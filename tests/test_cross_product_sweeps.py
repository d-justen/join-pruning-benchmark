from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import duckdb

from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.duckdb_runner import render_workload_sql
from benchmarking.experiment_config import (
    BenchmarkOptions,
    CrossProductExperimentDefinition,
    CrossProductFixedParameters,
    CrossProductQueryTypeConfig,
    CrossProductSweepConfig,
    CrossProductWorkloadConfig,
    VariantConfig,
    load_experiment_config,
)
from benchmarking.experiment_runner import run_experiment
from benchmarking.manifests import load_manifest, write_manifest
from benchmarking.query_inventory import load_queries
from benchmarking.results_analysis import _profile_sql
from benchmarking.workloads.cross_product import (
    _resolve_sweep_cases,
    _shoulder_preserving_key_expr,
    _spread_key_expr,
    prepare_cross_product_workload,
)


def make_spec(
    sweep_kind: str,
    values: tuple[int | float, ...],
    *,
    dimension_cardinality: int = 20,
    key_domain_size: int | str | None = None,
    selectivity: float = 0.5,
    join_count: int = 1,
    multiplicity: int = 1,
    probe_layout: str = "uniform",
    query_types: tuple[CrossProductQueryTypeConfig, ...] | None = None,
) -> CrossProductWorkloadConfig:
    if query_types is None and sweep_kind != "cluster_count":
        query_types = (
            CrossProductQueryTypeConfig(
                kind="range",
                cluster_count=None,
                probe_layouts=(probe_layout,),
            ),
        )
    definition = CrossProductExperimentDefinition(
        fixed=CrossProductFixedParameters(
            fact_rows=2_000,
            dimension_cardinality=dimension_cardinality,
            key_domain_size=key_domain_size,
            selectivity=selectivity,
            join_count=join_count,
            build_key_multiplicity=multiplicity,
            probe_layout=probe_layout,
            row_group_size=122_880,
            dummy_string_cols=1,
            dummy_string_len=8,
            seed=7,
        ),
        sweep=CrossProductSweepConfig(
            kind=sweep_kind,
            values=values,
            cardinality_mode=(
                "fixed_dimension_cardinality"
                if sweep_kind == "build_key_multiplicity"
                else None
            ),
        ),
        query_types=query_types or (),
    )
    return CrossProductWorkloadConfig(
        kind="cross_product",
        name=f"test-{sweep_kind}",
        build_sizes=(),
        selectivities=(),
        predicate_types=(),
        probe_clusterings=(),
        fact_rows=definition.fixed.fact_rows,
        dummy_string_cols=definition.fixed.dummy_string_cols,
        dummy_string_len=definition.fixed.dummy_string_len,
        seed=definition.fixed.seed,
        remote_http_parquet_latency_ms=None,
        experiment_definition=definition,
    )


class CrossProductSweepTest(unittest.TestCase):
    def test_paper_standardized_contract(self) -> None:
        config = load_experiment_config(Path(__file__).resolve().parents[1] / "paper/configs/local_paper_sweeps.json")
        self.assertFalse(config.overwrite_output)
        for workload in config.workloads:
            definition = workload.experiment_definition
            self.assertEqual(definition.fixed.dimension_cardinality, 100_000_000)
            self.assertEqual(definition.fixed.key_domain_size, 100_000_000)
            self.assertEqual(definition.fixed.selectivity, .01)
            cases = _resolve_sweep_cases(definition, storage_kind="duckdb")
            for case in cases:
                if definition.sweep.kind == "key_domain_size":
                    self.assertEqual(case.selected_build_rows, 1_000_000)
                    self.assertEqual(case.dimension_cardinality, case.key_domain_size)
                    self.assertEqual(case.probe_selectivity, .01)
                else:
                    self.assertEqual(case.key_domain_size, 100_000_000)
        for system in config.systems:
            for variant in system.variants:
                self.assertIn("SET disabled_optimizers='compressed_materialization';", variant.setup_sql)

    def test_fixed_density_preserves_matching_rows_and_reference_values(self) -> None:
        query_types = tuple(CrossProductQueryTypeConfig(kind, None, ("uniform",))
                            for kind in ("shoulder", "scattered"))
        base = make_spec("join_count", (1,), dimension_cardinality=1000,
                         key_domain_size=1000, selectivity=.1, query_types=query_types)
        density = replace(base, name="density", experiment_definition=replace(
            base.experiment_definition,
            sweep=CrossProductSweepConfig("key_domain_size", (200, 1000, 2000), "fixed_selected_build_and_probe")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = prepare_cross_product_workload(base, root / "baseline")
            varied = prepare_cross_product_workload(density, root / "density")
            con = duckdb.connect(str(varied.db_path), read_only=True)
            try:
                # ASOF complement lookups must not physically cluster the fact
                # table by a density FK; restore the reference row order.
                self.assertEqual(con.execute("SELECT fact_id FROM fact ORDER BY rowid").fetchall(),
                                 [(i,) for i in range(1, 2001)])
                con.execute(f"ATTACH '{baseline.db_path}' AS baseline (READ_ONLY)")
                anchor = con.execute("SELECT fact_id, fk_joins_j01_uniform FROM baseline.fact ORDER BY fact_id").fetchall()
                for kind in ("shoulder", "scattered"):
                    actual = con.execute(f"SELECT fact_id, fk_d1000_j01_uniform_{kind} FROM fact ORDER BY fact_id").fetchall()
                    self.assertEqual(actual, anchor)
                    expected = con.execute(f"""SELECT f.fact_id FROM baseline.fact f
                        JOIN baseline.dim_joins_j01 d ON f.fk_joins_j01_uniform=d.d_key
                        WHERE d.mask_s0100_{kind} ORDER BY f.fact_id""").fetchall()
                    for domain, mask in ((200, "s0500"), (1000, "s0100"), (2000, "s0050")):
                        self.assertEqual(con.execute(f"SELECT count(*) FROM dim_d{domain}_j01 WHERE mask_{mask}_{kind}").fetchone()[0], 100)
                        actual = con.execute(f"""SELECT f.fact_id FROM fact f
                            JOIN dim_d{domain}_j01 d ON f.fk_d{domain}_j01_uniform_{kind}=d.d_key
                            WHERE d.mask_{mask}_{kind} ORDER BY f.fact_id""").fetchall()
                        self.assertEqual(actual, expected)
                        self.assertEqual(con.execute(f"SELECT count(*) FROM fact WHERE fk_d{domain}_j01_uniform_{kind} IS NULL OR fk_d{domain}_j01_uniform_{kind} NOT BETWEEN 1 AND {domain}").fetchone()[0], 0)
                    self.assertEqual(con.execute(f"SELECT d_key,dummy_s1 FROM dim_d1000_j01 WHERE mask_s0100_{kind} ORDER BY d_key").fetchall(),
                                     con.execute(f"SELECT d_key,dummy_s1 FROM baseline.dim_joins_j01 WHERE mask_s0100_{kind} ORDER BY d_key").fetchall())
            finally:
                con.close()
            metadata = json.loads(varied.query_catalog_path.read_text())["queries"]
            self.assertTrue(all(q["probe_selectivity"] == .1 for q in metadata.values()))
            self.assertTrue(all(q["individual_join_selectivity"] == .1 for q in metadata.values()))
            self.assertTrue(all(q["expected_join_output_rows"] == 200 for q in metadata.values()))

    def test_fixed_domain_multiplicity_keeps_shoulders_and_independent_scattered_layout(self) -> None:
        queries = tuple(CrossProductQueryTypeConfig(kind, None, ("uniform",))
                        for kind in ("shoulder", "scattered"))
        spec = make_spec("build_key_multiplicity", (1, 2, 4), dimension_cardinality=1000,
                         key_domain_size=1000, selectivity=.1, query_types=queries)
        only_scattered = replace(spec, experiment_definition=replace(spec.experiment_definition, query_types=queries[1:]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare_cross_product_workload(spec, root / "both")
            other = prepare_cross_product_workload(only_scattered, root / "scattered")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                con.execute(f"ATTACH '{other.db_path}' AS other (READ_ONLY)")
                for multiplicity in (2, 4):
                    selected = round(1000 / multiplicity * .1)
                    lo, hi = selected // 2, selected - selected // 2
                    keys = con.execute(f"SELECT DISTINCT d_key FROM dim_m{multiplicity}_j01_shoulder WHERE mask_s0100_shoulder ORDER BY d_key").fetchall()
                    self.assertEqual([x[0] for x in keys], list(range(1, lo + 1)) + list(range(1001 - hi, 1001)))
                    for layout in ("shoulder", "spread"):
                        self.assertEqual(con.execute(f"SELECT count(*),count(DISTINCT d_key),min(d_key),max(d_key) FROM dim_m{multiplicity}_j01_{layout}").fetchone(),
                                         (1000, 1000 // multiplicity, 1, 1000))
                    table = f"dim_m{multiplicity}_j01_spread"
                    self.assertEqual(con.execute(f"SELECT DISTINCT d_key FROM {table} WHERE mask_s0100_scattered ORDER BY d_key").fetchall(),
                                     con.execute(f"SELECT DISTINCT d_key FROM other.{table} WHERE mask_s0100_scattered ORDER BY d_key").fetchall())
                    self.assertEqual(con.execute(f"SELECT fact_id,fk_m{multiplicity}_j01_uniform_spread FROM fact ORDER BY fact_id").fetchall(),
                                     con.execute(f"SELECT fact_id,fk_m{multiplicity}_j01_uniform_spread FROM other.fact ORDER BY fact_id").fetchall())
            finally:
                con.close()

    def test_common_sweep_points_reuse_keys_payloads_and_foreign_keys(self) -> None:
        queries = tuple(CrossProductQueryTypeConfig(kind, None, ("uniform",))
                        for kind in ("shoulder", "scattered"))
        reference = {}
        with tempfile.TemporaryDirectory() as directory:
            for sweep, value in (("join_count", 1), ("build_key_multiplicity", 1),
                                 ("selectivity", .1), ("row_group_size", 122880),
                                 ("cluster_count", 2)):
                spec = make_spec(sweep, (value,), dimension_cardinality=1000,
                                 key_domain_size=1000, selectivity=.1,
                                 query_types=None if sweep == "cluster_count" else queries)
                manifest = prepare_cross_product_workload(spec, Path(directory) / sweep)
                catalog = json.loads(manifest.query_catalog_path.read_text())["queries"]
                with duckdb.connect(str(manifest.db_path), read_only=True) as con:
                    for path in manifest.setup_sql_paths:
                        con.execute(path.read_text())
                    for query_name, parameters in catalog.items():
                        table = parameters["dimension_tables"][0]
                        kind = parameters["query_type"]
                        mask = "mask_s0100_seasonal_c00002" if kind == "seasonal" else f"mask_s0100_{kind}"
                        keys = con.execute(f"SELECT d_key,dummy_s1 FROM {table} WHERE {mask} ORDER BY d_key").fetchall()
                        # Execute the rendered join too, including attached row-group databases.
                        answer = con.execute((manifest.queries_dir / query_name).read_text()).fetchall()
                        fact_relation = "fact"
                        if sweep == "row_group_size":
                            fact_relation = "fact_rg122880.fact"
                        columns = con.execute(f"DESCRIBE {fact_relation}").fetchall()
                        fk = next(column[0] for column in columns if column[0].startswith("fk_"))
                        facts = con.execute(f"SELECT fact_id,{fk},measure FROM {fact_relation} ORDER BY fact_id").fetchall()
                        shape = "shoulder" if kind == "seasonal" else kind
                        actual = keys, facts, answer
                        if shape in reference:
                            self.assertEqual(actual, reference[shape], (sweep, shape))
                        else:
                            reference[shape] = actual

    def test_join_sweep_fixes_join_order_after_variant_setup(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "paper/configs/local_paper_sweeps.json"
        config = load_experiment_config(config_path)
        join_workload = next(workload for workload in config.workloads if workload.name == "jpb-sweep-joins")
        self.assertEqual(len(join_workload.post_variant_setup_sql), 1)
        self.assertIn("join_order,build_side_probe_side", join_workload.post_variant_setup_sql[0])
        self.assertTrue(all(
            not workload.post_variant_setup_sql
            for workload in config.workloads
            if workload.name != "jpb-sweep-joins"
        ))

        spec = replace(
            make_spec("join_count", (5,), dimension_cardinality=20),
            post_variant_setup_sql=join_workload.post_variant_setup_sql,
        )
        variant = VariantConfig(
            label="prf uncompressed",
            setup_sql=("SET disabled_optimizers='compressed_materialization';",),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = prepare_cross_product_workload(spec, root / "workload")
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, manifest)
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.post_variant_setup_sql, join_workload.post_variant_setup_sql)
            query = load_queries(manifest.queries_dir)[0]
            system = ResolvedSystem("binary", "duckdb", Path("duckdb"), "test", None, None, None)
            benchmark = BenchmarkOptions(1, None, "duckdb", False)
            sql = render_workload_sql(
                "test", manifest, system, variant, [query], root / "results.duckdb", benchmark,
                "run", root / "profiles", (),
            )
            setting = join_workload.post_variant_setup_sql[0]
            self.assertLess(sql.index(variant.setup_sql[0]), sql.index(setting))
            profile_sql = _profile_sql(
                query.path.read_text(encoding="utf-8"), root / "profile.json", variant, None, (),
                manifest.post_variant_setup_sql,
            )
            self.assertLess(profile_sql.index(variant.setup_sql[0]), profile_sql.index(setting))

            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                con.execute(variant.setup_sql[0])
                con.execute(setting)
                plan = json.loads(con.execute(
                    "EXPLAIN (FORMAT JSON) " + query.path.read_text(encoding="utf-8")
                ).fetchone()[1])[0]

                def find_join(node: dict) -> dict | None:
                    if node["name"] == "HASH_JOIN":
                        return node
                    for child in node.get("children", []):
                        found = find_join(child)
                        if found is not None:
                            return found
                    return None

                def scan_tables(node: dict) -> list[str]:
                    tables = []
                    if node["name"] == "SEQ_SCAN":
                        tables.append(node["extra_info"]["Table"])
                    for child in node.get("children", []):
                        tables.extend(scan_tables(child))
                    return tables

                join = find_join(plan)
                for join_index in (5, 4, 3, 2, 1):
                    self.assertIsNotNone(join)
                    probe, build = join["children"]
                    self.assertEqual(
                        [table.split(".")[-1] for table in scan_tables(build)],
                        [f"dim_joins_j{join_index:02d}"],
                    )
                    if join_index > 1:
                        join = find_join(probe)
                    else:
                        self.assertEqual(
                            [table.split(".")[-1] for table in scan_tables(probe)],
                            ["fact"],
                        )
            finally:
                con.close()

    def test_large_key_spread_is_exact_and_distinct(self) -> None:
        con = duckdb.connect()
        try:
            dense_expr = _spread_key_expr("key_rank", 100_000_000, 100_000_000)
            dense_rows = con.execute(
                f"SELECT key_rank, {dense_expr} FROM range(94_999_900, 95_000_100) t(key_rank)"
            ).fetchall()
            self.assertEqual(
                [key for rank, key in dense_rows],
                [rank + 1 for rank, _ in dense_rows],
            )

            sparse_expr = _spread_key_expr("key_rank", 80_000_000, 100_000_000)
            sparse_rows = con.execute(
                f"SELECT key_rank, {sparse_expr} FROM range(79_999_900, 80_000_000) t(key_rank)"
            ).fetchall()
            self.assertEqual(
                [key for rank, key in sparse_rows],
                [1 + (rank * 99_999_999) // 79_999_999 for rank, _ in sparse_rows],
            )
            self.assertEqual(len({key for _, key in sparse_rows}), len(sparse_rows))

            shoulder_expr = _shoulder_preserving_key_expr(
                "key_rank", 80_000_000, 100_000_000, 5_000_000, 5_000_000
            )
            shoulder_rows = con.execute(
                f"SELECT key_rank, {shoulder_expr} "
                "FROM (VALUES (0), (4999999), (5000000), (5000001), "
                "(74999998), (74999999), (75000000), (79999999)) t(key_rank)"
            ).fetchall()
            self.assertEqual(
                [key for _, key in shoulder_rows],
                [
                    rank + 1 if rank < 5_000_000 else
                    95_000_001 + rank - 75_000_000 if rank >= 75_000_000 else
                    5_000_001 + (rank - 5_000_000) * 89_999_999 // 69_999_999
                    for rank, _ in shoulder_rows
                ],
            )
            self.assertEqual(len({key for _, key in shoulder_rows}), len(shoulder_rows))
        finally:
            con.close()

    def test_migrated_paper_shape_matches_legacy_materialization(self) -> None:
        query_types = tuple(
            CrossProductQueryTypeConfig(
                kind,
                10 if kind == "seasonal" else None,
                ("ordered", "uniform"),
            )
            for kind in ("range", "shoulder", "seasonal", "scattered")
        )
        migrated = make_spec(
            "dimension_cardinality",
            (1_000, 10_000),
            selectivity=0.01,
            query_types=query_types,
        )
        legacy = CrossProductWorkloadConfig(
            kind="cross_product",
            name="legacy-paper-shape",
            build_sizes=(1_000, 10_000),
            selectivities=(0.01,),
            predicate_types=("range", "shoulder", "seasonal", "scattered"),
            probe_clusterings=("ordered", "uniform"),
            fact_rows=2_000,
            dummy_string_cols=1,
            dummy_string_len=8,
            seed=7,
            remote_http_parquet_latency_ms=None,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            migrated_manifest = prepare_cross_product_workload(migrated, root / "migrated")
            legacy_manifest = prepare_cross_product_workload(legacy, root / "legacy")
            self.assertEqual(
                {path.name for path in migrated_manifest.queries_dir.glob("*.sql")},
                {path.name for path in legacy_manifest.queries_dir.glob("*.sql")},
            )
            con = duckdb.connect()
            try:
                con.execute(f"ATTACH '{migrated_manifest.db_path}' AS migrated")
                con.execute(f"ATTACH '{legacy_manifest.db_path}' AS legacy")
                for table_name in ("fact", "dim_1k", "dim_10k"):
                    difference_count = con.execute(
                        f"""
                        SELECT count(*) FROM (
                            (SELECT * FROM migrated.{table_name}
                             EXCEPT ALL
                             SELECT * FROM legacy.{table_name})
                            UNION ALL
                            (SELECT * FROM legacy.{table_name}
                             EXCEPT ALL
                             SELECT * FROM migrated.{table_name})
                        ) differences
                        """
                    ).fetchone()[0]
                    self.assertEqual(difference_count, 0, table_name)
            finally:
                con.close()

    def test_cluster_count_sweep_rejects_one_cluster(self) -> None:
        spec = make_spec(
            "cluster_count",
            (1, 2),
            dimension_cardinality=20,
            selectivity=0.5,
            query_types=(),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "at least 2"):
                prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")

    def test_runner_persists_resolved_query_parameters(self) -> None:
        binary = shutil.which("duckdb")
        if binary is None:
            self.skipTest("DuckDB CLI is not installed")
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "sweep-smoke",
                        "output_root": str(root / "out"),
                        "overwrite_output": True,
                        "workloads": [
                            {
                                "name": "selectivity-sweep",
                                "kind": "cross_product",
                                "params": {
                                    "fixed": {
                                        "fact_rows": 2000,
                                        "dimension_cardinality": 20,
                                        "key_domain_size": "dimension_cardinality",
                                        "selectivity": 0.5,
                                        "join_count": 1,
                                        "build_key_multiplicity": 1,
                                        "probe_layout": "uniform",
                                        "row_group_size": 122880,
                                        "dummy_string_cols": 0,
                                        "seed": 7,
                                    },
                                    "sweep": {
                                        "kind": "selectivity",
                                        "values": [0.25, 0.5],
                                    },
                                    "query_types": [{"kind": "range"}],
                                },
                            }
                        ],
                        "systems": [
                            {
                                "name": "duckdb",
                                "kind": "binary",
                                "binary_path": binary,
                                "variants": [{"label": "default", "setup_sql": []}],
                            }
                        ],
                        "benchmark": {
                            "repetitions": 1,
                            "collect_profiles": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths = run_experiment(load_experiment_config(config_path))
            con = duckdb.connect(str(paths.results_db), read_only=True)
            try:
                rows = con.execute(
                    "SELECT query_name, query_parameters FROM timing_runs ORDER BY query_name"
                ).fetchall()
            finally:
                con.close()
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {json.loads(row[1])["individual_join_selectivity"] for row in rows},
                {0.25, 0.5},
            )

    def test_dimension_sweep_preserves_paper_query_names_and_writes_catalogs(self) -> None:
        query_types = tuple(
            CrossProductQueryTypeConfig(
                kind,
                10 if kind == "seasonal" else None,
                ("ordered", "uniform"),
            )
            for kind in ("range", "shoulder", "seasonal", "scattered")
        )
        spec = make_spec(
            "dimension_cardinality",
            (100, 1_000),
            selectivity=0.1,
            query_types=query_types,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact_dir = Path(temporary_dir) / "workload"
            manifest = prepare_cross_product_workload(spec, artifact_dir)

            query_names = {query.name for query in load_queries(manifest.queries_dir)}
            self.assertIn("b100_s0100_range_ordered.sql", query_names)
            self.assertIn("b1k_s0100_seasonal_uniform.sql", query_names)
            self.assertEqual(len(query_names), 16)
            self.assertTrue(manifest.definition_path.is_file())
            self.assertTrue(manifest.cases_path.is_file())
            self.assertTrue(manifest.query_catalog_path.is_file())
            definition = json.loads(manifest.definition_path.read_text(encoding="utf-8"))
            self.assertIn("parameter_descriptions", definition)
            self.assertEqual(
                set(definition["supported_experiment_configurations"]),
                {
                    "dimension_cardinality",
                    "selectivity",
                    "key_domain_size",
                    "build_key_multiplicity",
                    "join_count",
                    "cluster_count",
                    "row_group_size",
                },
            )
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                for query in load_queries(manifest.queries_dir):
                    con.execute(query.path.read_text(encoding="utf-8")).fetchall()
            finally:
                con.close()

    def test_density_sweep_uses_existing_sparse_dimension_keys_for_probe(self) -> None:
        spec = make_spec("key_domain_size", (20, 200), dimension_cardinality=20)
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                self.assertEqual(
                    con.execute("SELECT max(d_key) FROM dim_d200_j01_spread").fetchone()[0],
                    200,
                )
                missing = con.execute(
                    """
                    SELECT count(*)
                    FROM fact f
                    ANTI JOIN dim_d200_j01_spread d
                      ON f.fk_d200_j01_uniform_spread = d.d_key
                    """
                ).fetchone()[0]
                self.assertEqual(missing, 0)
            finally:
                con.close()

    def test_density_sweep_shoulders_are_contiguous_domain_boundary_ranges(self) -> None:
        query_types = (
            CrossProductQueryTypeConfig("shoulder", None, ("ordered",)),
            CrossProductQueryTypeConfig("scattered", None, ("ordered",)),
        )
        spec = make_spec(
            "key_domain_size",
            (100, 1_000),
            dimension_cardinality=100,
            key_domain_size=100,
            selectivity=0.1,
            probe_layout="ordered",
            query_types=query_types,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                selected_keys = con.execute(
                    """
                    SELECT d_key
                    FROM dim_d1000_j01_shoulder
                    WHERE mask_s0100_shoulder
                    ORDER BY d_key
                    """
                ).fetchall()
                self.assertEqual(
                    [row[0] for row in selected_keys],
                    [1, 2, 3, 4, 5, 996, 997, 998, 999, 1_000],
                )
                island_count = con.execute(
                    """
                    WITH selected AS (
                        SELECT
                            d_key,
                            d_key - row_number() OVER (ORDER BY d_key) AS island
                        FROM dim_d1000_j01_shoulder
                        WHERE mask_s0100_shoulder
                    )
                    SELECT count(DISTINCT island) FROM selected
                    """
                ).fetchone()[0]
                self.assertEqual(island_count, 2)

                dimension_counts = con.execute(
                    """
                    SELECT count(*), count(DISTINCT d_key)
                    FROM dim_d1000_j01_shoulder
                    """
                ).fetchone()
                self.assertEqual(dimension_counts, (100, 100))
                scattered_counts = con.execute(
                    """
                    WITH selected AS (
                        SELECT
                            d_key,
                            d_key - row_number() OVER (ORDER BY d_key) AS island
                        FROM dim_d1000_j01_spread
                        WHERE mask_s0100_scattered
                    )
                    SELECT count(*), count(DISTINCT island) FROM selected
                    """
                ).fetchone()
                self.assertEqual(scattered_counts[0], 10)
                self.assertGreater(scattered_counts[1], 2)

                joined_rows = con.execute(
                    """
                    SELECT count(*)
                    FROM fact f
                    JOIN dim_d1000_j01_shoulder d
                      ON f.fk_d1000_j01_ordered_shoulder = d.d_key
                    WHERE d.mask_s0100_shoulder
                    """
                ).fetchone()[0]
                self.assertEqual(joined_rows, 200)
            finally:
                con.close()

    def test_multiplicity_sweep_keeps_dimension_rows_fixed(self) -> None:
        spec = make_spec(
            "build_key_multiplicity",
            (1, 4),
            dimension_cardinality=20,
            key_domain_size=20,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                rows = con.execute(
                    """
                    SELECT count(*), count(DISTINCT d_key)
                    FROM dim_m4_j01_spread
                    """
                ).fetchone()
                self.assertEqual(rows, (20, 5))
                selected = con.execute(
                    """
                    SELECT count(*), count(DISTINCT d_key)
                    FROM dim_m4_j01_spread
                    WHERE mask_s0500_range
                    """
                ).fetchone()
                self.assertEqual(selected, (8, 2))
            finally:
                con.close()

    def test_multiplicity_can_keep_the_distinct_key_domain_dense(self) -> None:
        spec = make_spec(
            "build_key_multiplicity",
            (1, 4),
            dimension_cardinality=20,
            key_domain_size="dimension_distinct_keys",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            cases = json.loads(manifest.cases_path.read_text(encoding="utf-8"))["cases"]
            self.assertEqual(
                [(case["dimension_distinct_keys"], case["key_domain_size"]) for case in cases],
                [(20, 20), (5, 5)],
            )
            definition = json.loads(manifest.definition_path.read_text(encoding="utf-8"))
            self.assertEqual(
                definition["key_domain_size_mode"],
                "dimension_distinct_keys",
            )

    def test_join_count_sweep_generates_homogeneous_distinct_joins(self) -> None:
        spec = make_spec("join_count", (1, 3), dimension_cardinality=20)
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            query = (manifest.queries_dir / "j3_range_uniform.sql").read_text(encoding="utf-8")
            self.assertEqual(query.count("JOIN dim_joins_"), 3)
            catalog = json.loads(manifest.query_catalog_path.read_text(encoding="utf-8"))
            parameters = catalog["queries"]["j3_range_uniform.sql"]
            self.assertEqual(parameters["join_count"], 3)
            self.assertEqual(parameters["build_key_multiplicity"], 1)
            self.assertEqual(parameters["expected_query_selectivity"], 0.125)
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                actual_rows = con.execute(
                    """
                    SELECT count(*)
                    FROM fact f
                    JOIN dim_joins_j01 d1 ON f.fk_joins_j01_uniform = d1.d_key
                    JOIN dim_joins_j02 d2 ON f.fk_joins_j02_uniform = d2.d_key
                    JOIN dim_joins_j03 d3 ON f.fk_joins_j03_uniform = d3.d_key
                    WHERE d1.mask_s0500_range
                      AND d2.mask_s0500_range
                      AND d3.mask_s0500_range
                    """
                ).fetchone()[0]
                self.assertLess(abs(actual_rows / 2_000 - 0.125), 0.04)
                dimension_tables = con.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'dim_joins%'
                    ORDER BY table_name
                    """
                ).fetchall()
                self.assertEqual(
                    [row[0] for row in dimension_tables],
                    ["dim_joins_j01", "dim_joins_j02", "dim_joins_j03"],
                )
                fact_columns = [row[0] for row in con.execute("DESCRIBE fact").fetchall()]
                self.assertEqual(
                    [name for name in fact_columns if name.startswith("fk_joins")],
                    [
                        "fk_joins_j01_uniform",
                        "fk_joins_j02_uniform",
                        "fk_joins_j03_uniform",
                    ],
                )
            finally:
                con.close()

    def test_cluster_count_sweep_has_exact_separated_cluster_counts(self) -> None:
        spec = make_spec(
            "cluster_count",
            (2, 4),
            dimension_cardinality=20,
            selectivity=0.5,
            query_types=(),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                for cluster_count in (2, 4):
                    actual = con.execute(
                        f"""
                        WITH selected AS (
                            SELECT d_key,
                                   d_key - row_number() OVER (ORDER BY d_key) AS island
                            FROM dim_clusters_j01
                            WHERE mask_s0500_seasonal_c{cluster_count:05d}
                        )
                        SELECT count(DISTINCT island) FROM selected
                        """
                    ).fetchone()[0]
                    self.assertEqual(actual, cluster_count)
            finally:
                con.close()

    def test_large_cluster_count_uses_compact_set_based_generation(self) -> None:
        spec = make_spec(
            "cluster_count",
            (100_000,),
            dimension_cardinality=200_000,
            selectivity=0.5,
            query_types=(),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                selected = con.execute(
                    """
                    SELECT count(*)
                    FROM dim_clusters_j01
                    WHERE mask_s0500_seasonal_c100000
                    """
                ).fetchone()[0]
                self.assertEqual(selected, 100_000)
            finally:
                con.close()

    def test_selectivity_sweep_reuses_dimension_and_fact_storage(self) -> None:
        query_types = (
            CrossProductQueryTypeConfig("shoulder", None, ("ordered",)),
            CrossProductQueryTypeConfig("scattered", None, ("ordered",)),
        )
        spec = make_spec(
            "selectivity",
            (0.0001, 0.0003, 0.1),
            dimension_cardinality=10_000,
            selectivity=0.1,
            probe_layout="ordered",
            query_types=query_types,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = prepare_cross_product_workload(spec, Path(temporary_dir) / "workload")
            con = duckdb.connect(str(manifest.db_path), read_only=True)
            try:
                tables = con.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name LIKE 'dim_selectivity%'
                    """
                ).fetchall()
                self.assertEqual(tables, [("dim_selectivity_j01",)])
                columns = [row[0] for row in con.execute("DESCRIBE fact").fetchall()]
                self.assertEqual(
                    [name for name in columns if name.startswith("fk_")],
                    ["fk_selectivity_j01_ordered"],
                )
                dimension_columns = [
                    row[0] for row in con.execute("DESCRIBE dim_selectivity_j01").fetchall()
                ]
                self.assertIn("mask_s0p0001_shoulder", dimension_columns)
                self.assertIn("mask_s0p0003_scattered", dimension_columns)
                selected_counts = con.execute(
                    """
                    SELECT
                        count(*) FILTER (WHERE mask_s0p0001_scattered),
                        count(*) FILTER (WHERE mask_s0p0003_scattered),
                        count(*) FILTER (WHERE mask_s0100_scattered),
                        count(*) FILTER (
                            WHERE mask_s0p0001_scattered
                              AND NOT mask_s0p0003_scattered
                        )
                    FROM dim_selectivity_j01
                    """
                ).fetchone()
                self.assertEqual(selected_counts, (1, 3, 1_000, 0))
                for query in load_queries(manifest.queries_dir):
                    con.execute(query.path.read_text(encoding="utf-8")).fetchall()
            finally:
                con.close()

    def test_row_group_sweep_materializes_native_fact_databases(self) -> None:
        spec = make_spec("row_group_size", (2_048, 4_096), dimension_cardinality=20)
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact_dir = Path(temporary_dir) / "workload"
            manifest = prepare_cross_product_workload(spec, artifact_dir)
            cases = json.loads(manifest.cases_path.read_text(encoding="utf-8"))["cases"]
            self.assertEqual(
                [case["fact_storage"]["row_group_size"] for case in cases],
                [2_048, 4_096],
            )
            self.assertGreater(cases[0]["fact_storage"]["row_group_count"], 0)
            self.assertTrue((artifact_dir / "fact_rg2048.duckdb").is_file())
            self.assertIn(
                'FROM "fact_rg2048".fact AS f',
                (manifest.queries_dir / "rg2048_range_uniform.sql").read_text(
                    encoding="utf-8"
                ),
            )
            con = duckdb.connect(str(manifest.db_path))
            try:
                for setup_path in manifest.setup_sql_paths:
                    con.execute(setup_path.read_text(encoding="utf-8"))
                mismatches = con.execute(
                    """
                    SELECT count(*)
                    FROM fact_rg2048.fact a
                    JOIN fact_rg4096.fact b USING (fact_id)
                    WHERE a.fk_j01_uniform != b.fk_j01_uniform
                    """
                ).fetchone()[0]
                self.assertEqual(mismatches, 0)
                for query in load_queries(manifest.queries_dir):
                    con.execute(query.path.read_text(encoding="utf-8")).fetchall()
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
