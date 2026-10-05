#!/usr/bin/env python3
"""Small engine parity checks; never prepares or runs the paper sweep.

Compare filter policy with explicit statistics, then compare filtered/unfiltered
SQL answers and filter decisions. Source NDV estimates may differ by version.
Temporary databases/profiles are removed on exit; no source identity is recorded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def run(binary: Path, sql: str) -> list[dict]:
    result = subprocess.run([str(binary), "-batch", "-bail", "-json"], input=sql,
                            text=True, capture_output=True, check=True, timeout=180)
    # Each SELECT emits its own JSON array.
    decoder = json.JSONDecoder()
    output = result.stdout.strip()
    rows = []
    while output:
        data, end = decoder.raw_decode(output)
        rows.extend(data)
        output = output[end:].lstrip()
    return rows


def setup_sql(kind: str, selected: int, multiplicity: int) -> str:
    if kind == "shoulder":
        key = f"CASE WHEN r < {selected // 2} THEN r ELSE 100000000 - {selected} + r END"
    else:
        key = "r * 700"
    # Four source keys per selected key keep RPT's selectivity heuristic active.
    # All duplicates have the same predicate. Filler keys lie inside the span.
    return f"""
        CREATE TABLE b AS
        SELECT k::BIGINT k, keep FROM (
            SELECT {key} k, true keep FROM range({selected}) t(r)
            UNION ALL
            SELECT (30000000 + r)::BIGINT k, false keep FROM range({selected * 3}) t(r)
        ) keys CROSS JOIN range({multiplicity}) duplicates(i)
        ORDER BY hash(k), i;
        CREATE TABLE p AS
        SELECT DISTINCT k FROM b WHERE keep
        UNION ALL SELECT i::BIGINT k FROM range(0, 100000000, 997) t(i)
        UNION ALL SELECT NULL::BIGINT;
        CHECKPOINT;
    """


def check(binary: Path, engine: str, setup: str, threads: int, profile: Path) -> tuple[dict, dict]:
    common = f"""
        SET threads={threads};
        SET disabled_optimizers='compressed_materialization,join_order,build_side_probe_side';
        SET dynamic_or_filter_threshold=0;
    """
    if engine == "prf":
        enabled = """
            SET enable_prefix_range_filter=true;
            SET enable_prefix_range_filter_compression=true;
            SET enable_join_bloom_filter_pushdown=true;
            SET enable_join_min_max_filter_pushdown=true;
            SET enable_perfect_hash_join_filter_pushdown=false;
        """
        disabled = """
            SET enable_prefix_range_filter=false;
            SET enable_join_bloom_filter_pushdown=false;
            SET enable_join_min_max_filter_pushdown=false;
        """
    else:
        enabled = "SET rpt_filter='prf'; SET rpt_prf_scan_pushdown=true;"
        disabled = "SET rpt_filter='none';"
    query = "SELECT count(*) AS n, sum(p.k)::HUGEINT AS checksum FROM p JOIN b USING(k) WHERE b.keep;"
    escaped = str(profile).replace("'", "''")
    answers = run(binary, common + setup + enabled + f"""
        SET enable_profiling='json'; SET profiling_output='{escaped}';
        {query}
        SET enable_profiling='no_output';
    """ + disabled + query)
    # Older shells emit HUGEINT as a JSON number; newer ones use a string.
    answers = [{key: int(value) for key, value in row.items()} for row in answers]
    if len(answers) != 2 or answers[0] != answers[1]:
        raise AssertionError(f"{engine}: filtered/unfiltered answer mismatch: {answers}")
    data = json.loads(profile.read_text())
    if engine == "prf":
        metrics = [n["prefix_range_filter"] for n in walk(data) if "prefix_range_filter" in n]
        if len(metrics) != 1:
            raise AssertionError(f"Expected one regular PRF profile, got {len(metrics)}")
        m = metrics[0]
        result = {"mode": "direct_ranges" if m["final_prf_mode"] else "bitmap",
                  "fallback": bool(m["bloom_fallback_selected"]),
                  "fpr": float(m["final_prf_false_positive_rate"])}
    else:
        metrics = [n for n in walk(data) if "PRF 0 final_mode" in n]
        if len(metrics) != 1:
            raise AssertionError(f"Expected one RPT PRF profile, got {len(metrics)}")
        m = metrics[0]
        result = {"mode": m["PRF 0 final_mode"],
                  "fallback": m["PRF 0 bloom_fallback_selected"] == "true",
                  "fpr": float(m["PRF 0 false_positive_rate"])}
    return answers[0], result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prf", type=Path, required=True)
    parser.add_argument("--rpt", type=Path, required=True)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    fixtures = [("shoulder", k, 1) for k in (10000, 100000, 300000)]
    fixtures += [("shoulder", 12500, 8), ("scattered", 100000, 2)]
    with tempfile.TemporaryDirectory(prefix="prf-parity-") as directory:
        root = Path(directory)
        filter_rows = []
        for engine, binary in (("prf", args.prf), ("rpt", args.rpt)):
            binary = binary.resolve()
            source = binary.parents[2]
            library = binary.parent / "src"
            executable = root / f"{engine}-filter-fixture"
            command = [args.cxx, "-std=c++17", "-O2", "-I" + str(source / "src/include"),
                       str(Path(__file__).parent / "validation/prf_filter_fixture.cpp"),
                       "-L" + str(library), "-Wl,-rpath," + str(library), "-lduckdb", "-o", str(executable)]
            if engine == "rpt":
                command.append("-DDUCKDB_RPT")
            subprocess.run(command, check=True)
            output = subprocess.check_output([str(executable)], text=True, timeout=180)
            filter_rows.append([line.split("\t") for line in output.splitlines()])
        if len(filter_rows[0]) != len(filter_rows[1]):
            raise AssertionError("Filter fixture count differs")
        for regular, rpt in zip(*filter_rows):
            if (regular[:6] != rpt[:6] or regular[7:] != rpt[7:]
                    or not math.isclose(float(regular[6]), float(rpt[6]), rel_tol=1e-12, abs_tol=1e-15)):
                raise AssertionError(f"Explicit-statistics filter mismatch: {regular}, {rpt}")
        print(f"PASS {len(filter_rows[0])} filter fixtures: modes, shifts, ranges, active buckets, FPR, fallback, membership")
        for kind, selected, multiplicity in fixtures:
            # Prepare once in a storage format both versions can read. Both
            # engines then see the same physical rows and persisted statistics.
            shared = root / f"{kind}-{selected}-{multiplicity}.duckdb"
            escaped = str(shared).replace("'", "''")
            run(args.prf.resolve(), f"ATTACH '{escaped}' AS fixture (STORAGE_VERSION 'v1.2.0'); "
                "USE fixture; SET threads=1; " + setup_sql(kind, selected, multiplicity))
            setup = f"ATTACH '{escaped}' AS fixture (READ_ONLY); USE fixture;"
            previous = None
            for threads in (1, 4):
                regular = check(args.prf.resolve(), "prf", setup, threads, root / "prf.json")
                rpt = check(args.rpt.resolve(), "rpt", setup, threads, root / "rpt.json")
                label = f"{kind}, keys={selected}, rows/key={multiplicity}, threads={threads}"
                if regular[0] != rpt[0]:
                    raise AssertionError(f"{label}: engines disagree on join answer: {regular[0]}, {rpt[0]}")
                a, b = regular[1], rpt[1]
                if (a["mode"] != b["mode"] or a["fallback"] != b["fallback"]
                        or not math.isclose(a["fpr"], b["fpr"], rel_tol=1e-12, abs_tol=1e-15)):
                    raise AssertionError(f"{label}: PRF decision differs: regular={a}, rpt={b}")
                invariant = regular, rpt
                if previous is not None and invariant != previous:
                    raise AssertionError(f"{label}: serial/parallel result differs")
                if kind == "shoulder" and (b["mode"] != "direct_ranges" or b["fallback"]):
                    raise AssertionError(f"{label}: shoulder should use direct ranges")
                if kind == "scattered" and not b["fallback"]:
                    raise AssertionError(f"{label}: sparse keys should select Bloom fallback")
                previous = invariant
                print(f"PASS {label}: {b['mode']}, fallback={b['fallback']}; "
                      f"source-statistics FPR regular={a['fpr']:.8g}, rpt={b['fpr']:.8g}")


if __name__ == "__main__":
    main()
