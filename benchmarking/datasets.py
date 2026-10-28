"""Pinned inputs and generated external databases for the experiment suite.

Cache directories become reusable only after loading and validation succeed.
No path outside the managed cache is removed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import duckdb

from benchmarking.resources import connect
from benchmarking.suite_io import fingerprint, lock, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
ADAPTER_VERSION = 1


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def download(spec: dict, destination: Path) -> Path:
    """Validate even cached inputs; a partial download can never be a cache hit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock(destination.with_suffix(destination.suffix + ".lock")):
        if destination.is_file() and sha256(destination) == spec["sha256"]:
            return destination
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(3):
            partial.unlink(missing_ok=True)
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                 "--max-time", "3600", "--silent", "--show-error",
                 spec["url"], "--output", str(partial)], check=True,
            )
            if sha256(partial) == spec["sha256"] and partial.stat().st_size == spec["bytes"]:
                partial.replace(destination)
                return destination
            partial.unlink(missing_ok=True)
            time.sleep(attempt + 1)
        raise ValueError(f"checksum mismatch downloading {spec['url']}")


def input_lock() -> dict:
    return json.loads((ROOT / "docker/datasets.lock.json").read_text())


def airports(cache: Path) -> Path:
    spec = input_lock()["files"]["airports.csv"]
    return download(spec, cache / "downloads" / "airports.csv")


def recipe(spec: dict) -> dict:
    kind = spec["kind"]
    if kind not in ("job", "tpch", "ssb"):
        raise ValueError(f"unsupported dataset kind: {kind}")
    if kind in ("tpch", "ssb") and spec["scale_factor"] <= 0:
        raise ValueError("scale_factor must be positive")
    if kind == "ssb" and spec.get("order", "original") not in ("original", "ordered"):
        raise ValueError("SSB order must be original or ordered")
    result = {"adapter_version": ADAPTER_VERSION, "adapter_sha256": sha256(Path(__file__)),
              "duckdb_python": duckdb.__version__, "storage_compatibility_version": "v1.2.0",
              "parameters": spec}
    if kind == "job":
        result["inputs"] = {k: v for k, v in input_lock()["files"].items() if k.startswith("job_")}
        result["schema"] = sha256(ROOT / "external_benchmarks/join-order-benchmark/schema.sql")
    else:
        prefix = Path(spec.get("generator_root", "/opt/tpcdskew" if kind == "tpch" else "/opt/ssb"))
        result["generator_sha256"] = sha256(prefix / "bin/dbgen")
        distribution = prefix / ("share/tpcdskew/dists.dss" if kind == "tpch" else "share/dists.dss")
        result["distribution_sha256"] = sha256(distribution)
        schema = (ROOT / "external_benchmarks/tpch-skew/schema.sql" if kind == "tpch"
                  else prefix / "share/ssb.ddl")
        result["schema"] = sha256(schema)
    return result


def dataset_path(cache: Path, spec: dict) -> Path:
    return cache / "datasets" / fingerprint(recipe(spec))


def valid_dataset(path: Path, expected: dict) -> bool:
    marker = path / "complete.json"
    if not marker.is_file():
        return False
    info = json.loads(marker.read_text())
    database = path / "benchmark.duckdb"
    if info["recipe"] != expected or not database.is_file() or database.stat().st_size != info["bytes"]:
        return False
    if sha256(database) != info["sha256"]:
        return False
    # Row counts also verify that the database can be opened after interruption.
    with connect(str(database), read_only=True) as con:
        return all(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == rows
                   for table, rows in info["rows"].items())


def _validate(con, kind: str, scale: float | None) -> dict:
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1"
    ).fetchall()]
    counts = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables}
    if any(n <= 0 for n in counts.values()):
        raise ValueError(f"empty generated/input table: {counts}")
    if kind == "tpch":
        expected = {"region": 5, "nation": 25, "part": int(200000 * scale),
                    "supplier": int(10000 * scale), "customer": int(150000 * scale),
                    "orders": int(1500000 * scale), "partsupp": int(800000 * scale)}
        for table, rows in expected.items():
            if counts.get(table) != rows:
                raise ValueError(f"{table}: expected {rows}, found {counts.get(table)}")
        orphan_sql = [
            "SELECT count(*) FROM lineitem l ANTI JOIN orders o ON l.l_orderkey=o.o_orderkey",
            "SELECT count(*) FROM orders o ANTI JOIN customer c ON o.o_custkey=c.c_custkey",
            "SELECT count(*) FROM partsupp ps ANTI JOIN part p ON ps.ps_partkey=p.p_partkey",
            "SELECT count(*) FROM partsupp ps ANTI JOIN supplier s ON ps.ps_suppkey=s.s_suppkey",
        ]
    elif kind == "ssb":
        if len(counts) != 5 or counts.get("date") != 2557:
            raise ValueError(f"unexpected SSB tables: {counts}")
        orphan_sql = [
            f'SELECT count(*) FROM lineorder l ANTI JOIN "{table}" d ON l.lo_{fk}=d.{pk}'
            for table, fk, pk in [("date", "orderdate", "d_datekey"),
                                  ("customer", "custkey", "c_custkey"),
                                  ("supplier", "suppkey", "s_suppkey"),
                                  ("part", "partkey", "p_partkey")]
        ]
    else:
        if len(counts) != 21:
            raise ValueError(f"expected 21 JOB tables, found {len(counts)}")
        orphan_sql = []  # The historical IMDB snapshot itself contains dangling references.
    for sql in orphan_sql:
        if con.execute(sql).fetchone()[0]:
            raise ValueError(f"referential validation failed: {sql}")
    return counts


def prepare_dataset(cache: Path, spec: dict, *, existing_only: bool = False) -> Path:
    identity = recipe(spec)
    target = cache / "datasets" / fingerprint(identity)
    with lock(cache / "locks" / (target.name + ".lock")):
        if valid_dataset(target, identity):
            return target
        if existing_only:
            raise FileNotFoundError(f"dataset is not prepared: {spec}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        kind = spec["kind"]
        commands = []
        try:
            with connect(str(target / "benchmark.duckdb")) as con:
                con.execute("SET preserve_insertion_order=true")
                if kind == "job":
                    con.execute((ROOT / "external_benchmarks/join-order-benchmark/schema.sql").read_text())
                    for name, source in input_lock()["files"].items():
                        if not name.startswith("job_"):
                            continue
                        file = download(source, cache / "downloads" / name)
                        table = name.removeprefix("job_").removesuffix(".parquet")
                        con.execute(f'INSERT INTO "{table}" SELECT * FROM read_parquet({literal(file)})')
                elif kind in ("tpch", "ssb"):
                    prefix = Path(spec.get("generator_root", "/opt/tpcdskew" if kind == "tpch" else "/opt/ssb"))
                    raw = target / "raw"
                    raw.mkdir()
                    distribution = prefix / ("share/tpcdskew" if kind == "tpch" else "share")
                    command = [str(prefix / "bin/dbgen"), "-s", str(spec["scale_factor"])]
                    if kind == "tpch":
                        command += ["-z", str(spec.get("skew", 2))]
                    commands.append(command)
                    subprocess.run(command, cwd=raw, env={**os.environ, "DSS_PATH": str(raw),
                                   "DSS_CONFIG": str(distribution)}, check=True, stdin=subprocess.DEVNULL)
                    if kind == "ssb" and spec["scale_factor"] < 1:
                        # This dbgen scales DATE down for fractional SF even
                        # though fact dates still cover the full calendar.
                        # Restore the canonical fixed dimension via its own CLI.
                        date_command = [str(prefix / "bin/dbgen"), "-s", "1", "-T", "d", "-f"]
                        commands.append(date_command)
                        subprocess.run(date_command, cwd=raw, env={**os.environ, "DSS_PATH": str(raw),
                                       "DSS_CONFIG": str(distribution)}, check=True, stdin=subprocess.DEVNULL)
                    schema = (ROOT / "external_benchmarks/tpch-skew/schema.sql" if kind == "tpch"
                              else prefix / "share/ssb.ddl")
                    con.execute(schema.read_text())
                    tables = ([("region", "region"), ("nation", "nation"), ("supplier", "supplier"),
                               ("customer", "customer"), ("part", "part"), ("partsupp", "partsupp"),
                               ("orders", "order"), ("lineitem", "lineitem")] if kind == "tpch" else
                              [(t, t) for t in ("part", "supplier", "customer", "date", "lineorder")])
                    for table, filename in tables:
                        con.execute(f'COPY "{table}" FROM {literal(raw / (filename + ".tbl"))} (DELIMITER \'|\')')
                    if kind == "ssb" and spec.get("order", "original") == "ordered":
                        con.execute("CREATE TABLE sorted_lineorder AS SELECT * FROM lineorder ORDER BY lo_orderdate, rowid")
                        con.execute("DROP TABLE lineorder")
                        con.execute("ALTER TABLE sorted_lineorder RENAME TO lineorder")
                else:
                    raise ValueError(f"unsupported dataset kind: {kind}")
                rows = _validate(con, kind, spec.get("scale_factor"))
                con.execute("CHECKPOINT")
            if (target / "raw").exists():
                shutil.rmtree(target / "raw")
            write_json(target / "complete.json", {
                "recipe": identity, "rows": rows, "commands": commands,
                "bytes": (target / "benchmark.duckdb").stat().st_size,
                "sha256": sha256(target / "benchmark.duckdb"),
                "order": spec.get("order", "generator"), "status": "complete",
            })
        except BaseException:
            # Preserve logs in the unit directory; incomplete cache data is disposable.
            shutil.rmtree(target)
            raise
        return target
