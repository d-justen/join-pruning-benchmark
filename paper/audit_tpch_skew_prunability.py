from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "external_benchmarks/tpch-skew/tpch_sf100_z2.duckdb"
DEFAULT_PROFILES = (
    ROOT
    / "out/experiments/paper-local-ducklake-http-10ms/profiles/tpch-skew"
    / "duckdb-paper-ducklake-http"
)
DEFAULT_OUTPUT = ROOT / "out/audits/tpch-skew-prunability"

MIN_MAX_PATTERN = re.compile(r"\[Min: (-?\d+), Max: (-?\d+)\]")
BLOOM_ENUMERATION_SPAN = 2048


@dataclass(frozen=True)
class AuditEdge:
    query: str
    edge: str
    probe_table: str
    probe_column: str
    build_sql: str


# These are direct selective dimension-to-fact edges whose build-key set can be
# reconstructed without executing a large fact-side subquery. More complex
# fact-derived runtime filters are still represented in the profile summary.
AUDIT_EDGES = (
    AuditEdge(
        "q02",
        "part -> partsupp",
        "partsupp",
        "ps_partkey",
        "SELECT p_partkey AS build_key FROM part "
        "WHERE p_size = 15 AND p_type LIKE '%BRASS'",
    ),
    AuditEdge(
        "q02",
        "supplier -> partsupp",
        "partsupp",
        "ps_suppkey",
        "SELECT s_suppkey AS build_key FROM supplier "
        "JOIN nation ON s_nationkey = n_nationkey "
        "JOIN region ON n_regionkey = r_regionkey WHERE r_name = 'EUROPE'",
    ),
    AuditEdge(
        "q03",
        "customer -> orders",
        "orders",
        "o_custkey",
        "SELECT c_custkey AS build_key FROM customer "
        "WHERE c_mktsegment = 'BUILDING'",
    ),
    AuditEdge(
        "q05",
        "customer -> orders",
        "orders",
        "o_custkey",
        "SELECT c_custkey AS build_key FROM customer "
        "JOIN nation ON c_nationkey = n_nationkey "
        "JOIN region ON n_regionkey = r_regionkey WHERE r_name = 'ASIA'",
    ),
    AuditEdge(
        "q08",
        "part -> lineitem",
        "lineitem",
        "l_partkey",
        "SELECT p_partkey AS build_key FROM part "
        "WHERE p_type = 'ECONOMY ANODIZED STEEL'",
    ),
    AuditEdge(
        "q09",
        "part -> lineitem",
        "lineitem",
        "l_partkey",
        "SELECT p_partkey AS build_key FROM part WHERE p_name LIKE '%green%'",
    ),
    AuditEdge(
        "q11",
        "supplier -> partsupp",
        "partsupp",
        "ps_suppkey",
        "SELECT s_suppkey AS build_key FROM supplier "
        "JOIN nation ON s_nationkey = n_nationkey WHERE n_name = 'GERMANY'",
    ),
    AuditEdge(
        "q16",
        "part -> partsupp",
        "partsupp",
        "ps_partkey",
        "SELECT p_partkey AS build_key FROM part "
        "WHERE p_brand <> 'Brand#45' "
        "AND p_type NOT LIKE 'MEDIUM POLISHED%' "
        "AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)",
    ),
    AuditEdge(
        "q17",
        "part -> lineitem",
        "lineitem",
        "l_partkey",
        "SELECT p_partkey AS build_key FROM part "
        "WHERE p_brand = 'Brand#23' AND p_container = 'MED BOX'",
    ),
    AuditEdge(
        "q19",
        "part -> lineitem",
        "lineitem",
        "l_partkey",
        "SELECT p_partkey AS build_key FROM part "
        "WHERE p_brand IN ('Brand#12', 'Brand#23', 'Brand#34') "
        "AND p_container IN ("
        "'SM CASE', 'SM BOX', 'SM PACK', 'SM PKG', "
        "'MED BAG', 'MED BOX', 'MED PKG', 'MED PACK', "
        "'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')",
    ),
    AuditEdge(
        "q20",
        "part -> partsupp",
        "partsupp",
        "ps_partkey",
        "SELECT p_partkey AS build_key FROM part WHERE p_name LIKE 'forest%'",
    ),
    AuditEdge(
        "q21",
        "supplier -> lineitem",
        "lineitem",
        "l_suppkey",
        "SELECT s_suppkey AS build_key FROM supplier "
        "JOIN nation ON s_nationkey = n_nationkey "
        "WHERE n_name = 'SAUDI ARABIA'",
    ),
)


@dataclass(frozen=True)
class Interval:
    interval_id: int
    rows: int
    lower: int
    upper: int


@dataclass(frozen=True)
class EdgeResult:
    query: str
    edge: str
    probe_table: str
    probe_column: str
    granularity: str
    target_rows: int | None
    build_key_count: int
    build_key_min: int
    build_key_max: int
    intervals: int
    rows: int
    median_key_span: int
    intervals_bloom_enumerable: int
    minmax_survivors: int
    bloom_oracle_survivors: int
    range_oracle_survivors: int
    minmax_rows: int
    bloom_oracle_rows: int
    range_oracle_rows: int
    range_survivor_runs: int


def extract_row_group_intervals(
    con: duckdb.DuckDBPyConnection, table: str, column: str
) -> list[Interval]:
    rows = con.execute(
        f"""
        SELECT row_group_id, start, count, stats
        FROM pragma_storage_info('{table}')
        WHERE column_name = ? AND segment_type <> 'VALIDITY'
        ORDER BY row_group_id, start
        """,
        [column],
    ).fetchall()
    combined: dict[int, list[int]] = {}
    for row_group_id, start, count, stats in rows:
        match = MIN_MAX_PATTERN.search(stats)
        if not match:
            continue
        lower, upper = (int(value) for value in match.groups())
        row_count = start + count
        previous = combined.get(row_group_id)
        if previous is None:
            combined[row_group_id] = [row_count, lower, upper]
        else:
            previous[0] = max(previous[0], row_count)
            previous[1] = min(previous[1], lower)
            previous[2] = max(previous[2], upper)
    return [
        Interval(index, values[0], values[1], values[2])
        for index, (_, values) in enumerate(sorted(combined.items()))
    ]


def bundle_intervals(intervals: Iterable[Interval], target_rows: int) -> list[Interval]:
    result: list[Interval] = []
    rows = 0
    lower: int | None = None
    upper: int | None = None
    for interval in intervals:
        if rows and rows + interval.rows > target_rows:
            assert lower is not None and upper is not None
            result.append(Interval(len(result), rows, lower, upper))
            rows = 0
            lower = None
            upper = None
        rows += interval.rows
        lower = interval.lower if lower is None else min(lower, interval.lower)
        upper = interval.upper if upper is None else max(upper, interval.upper)
    if rows:
        assert lower is not None and upper is not None
        result.append(Interval(len(result), rows, lower, upper))
    return result


def survivor_runs(survivors: list[int]) -> int:
    return sum(
        1 for index, value in enumerate(survivors) if index == 0 or value != survivors[index - 1] + 1
    )


def audit_edge(
    con: duckdb.DuckDBPyConnection,
    edge: AuditEdge,
    intervals: list[Interval],
    granularity: str,
    target_rows: int | None,
) -> EdgeResult:
    con.execute("CREATE OR REPLACE TEMP TABLE audit_build_keys AS " + edge.build_sql)
    key_count, key_min, key_max = con.execute(
        "SELECT count(DISTINCT build_key), min(build_key), max(build_key) "
        "FROM audit_build_keys"
    ).fetchone()
    if key_count == 0:
        raise ValueError(f"no build keys selected for {edge.query} {edge.edge}")

    con.execute(
        "CREATE OR REPLACE TEMP TABLE audit_intervals("
        "interval_id INTEGER, rows BIGINT, lower BIGINT, upper BIGINT)"
    )
    con.executemany(
        "INSERT INTO audit_intervals VALUES (?, ?, ?, ?)",
        [(i.interval_id, i.rows, i.lower, i.upper) for i in intervals],
    )
    exact_rows = con.execute(
        """
        SELECT DISTINCT interval_id
        FROM audit_intervals i
        JOIN audit_build_keys k ON k.build_key BETWEEN i.lower AND i.upper
        ORDER BY interval_id
        """
    ).fetchall()
    exact_survivors = [row[0] for row in exact_rows]
    exact_set = set(exact_survivors)
    minmax_set = {
        interval.interval_id
        for interval in intervals
        if interval.upper >= key_min and interval.lower <= key_max
    }
    # This is optimistic for Bloom: false positives are ignored. Intervals too
    # wide for exhaustive enumeration conservatively survive, matching DuckDB.
    bloom_set = {
        interval.interval_id
        for interval in intervals
        if interval.upper - interval.lower >= BLOOM_ENUMERATION_SPAN
        or interval.interval_id in exact_set
    }
    row_counts = {interval.interval_id: interval.rows for interval in intervals}
    spans = sorted(interval.upper - interval.lower for interval in intervals)
    return EdgeResult(
        query=edge.query,
        edge=edge.edge,
        probe_table=edge.probe_table,
        probe_column=edge.probe_column,
        granularity=granularity,
        target_rows=target_rows,
        build_key_count=key_count,
        build_key_min=key_min,
        build_key_max=key_max,
        intervals=len(intervals),
        rows=sum(row_counts.values()),
        median_key_span=spans[len(spans) // 2],
        intervals_bloom_enumerable=sum(
            interval.upper - interval.lower < BLOOM_ENUMERATION_SPAN
            for interval in intervals
        ),
        minmax_survivors=len(minmax_set),
        bloom_oracle_survivors=len(bloom_set),
        range_oracle_survivors=len(exact_set),
        minmax_rows=sum(row_counts[i] for i in minmax_set),
        bloom_oracle_rows=sum(row_counts[i] for i in bloom_set),
        range_oracle_rows=sum(row_counts[i] for i in exact_set),
        range_survivor_runs=survivor_runs(exact_survivors),
    )


def walk_profile(node: dict) -> Iterable[dict]:
    yield node
    for child in node.get("children", []):
        yield from walk_profile(child)


def profile_summary(profiles_dir: Path) -> list[dict]:
    variants = {
        "no_prefilter": "no_prefilter",
        "sip": "sip",
        "sip_min_max": "sip_min_max",
        "prf": "prf_with_fallback",
    }
    result: list[dict] = []
    for query_number in range(1, 23):
        query = f"q{query_number:02d}"
        row: dict[str, object] = {"query": query}
        prf_targets: set[str] = set()
        for label, directory in variants.items():
            path = profiles_dir / directory / f"{query}.json"
            if not path.exists():
                row[f"{label}_scan_rows"] = None
                continue
            profile = json.loads(path.read_text(encoding="utf-8"))
            scan_rows = 0
            for node in walk_profile(profile):
                if (node.get("operator_name") or "").strip() not in {
                    "SEQ_SCAN",
                    "TABLE_SCAN",
                }:
                    continue
                scan_rows += int(node.get("operator_rows_scanned") or 0)
                if label == "prf":
                    extra = node.get("extra_info", {})
                    dynamic_filters = str(extra.get("Dynamic Filters", ""))
                    if "PRF(" in dynamic_filters:
                        table = str(extra.get("Table", "")).split(".")[-1]
                        prf_targets.add(table)
            row[f"{label}_scan_rows"] = scan_rows
        row["prf_targets"] = ", ".join(sorted(prf_targets))
        row["has_prf"] = bool(prf_targets)
        result.append(row)
    return result


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2f}x"


def percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "n/a"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    db_path: Path,
    target_rows: int,
    results: list[EdgeResult],
    profiles: list[dict],
) -> None:
    file_results = [result for result in results if result.granularity == "file_proxy"]
    promising = sorted(
        file_results,
        key=lambda result: -min(
            result.minmax_survivors / result.range_oracle_survivors,
            result.bloom_oracle_survivors / result.range_oracle_survivors,
        ),
    )
    best = promising[0]
    lines = [
        "# TPC-H Skew prunability audit",
        "",
        f"- Database: `{db_path}`",
        "- Dataset: TPC-H SF100, Zipf 2 (from the external workload metadata)",
        f"- File proxy: consecutive native row groups bundled up to {target_rows:,} rows",
        f"- Bloom rule: enumerate only when `max - min < {BLOOM_ENUMERATION_SPAN}`",
        "- Bloom oracle ignores hash false positives and is therefore optimistic",
        "- Range oracle is the best possible interval result for the exact build-key set; an actual PRF may retain extra false-positive intervals",
        "",
        "## Findings",
        "",
        f"The strongest dual-comparison candidate is **{best.query} {best.edge}**: min/max retains {best.minmax_survivors}/{best.intervals} file proxies, the optimistic Bloom oracle retains {best.bloom_oracle_survivors}/{best.intervals}, and the exact range oracle retains {best.range_oracle_survivors}/{best.intervals} in {best.range_survivor_runs} separated runs.",
        "",
        "Q08 and Q09 are excellent demonstrations against min/max but not against Bloom: their narrow/constant join-key intervals let Bloom enumerate the interval and match the exact range oracle. Q21 is the opposite: it strongly separates ranges from Bloom, but min/max is already optimal. Q17, Q19, and Q16 offer smaller improvements over both comparators.",
        "",
        "## Main candidates",
        "",
        "| Query edge | Files | Min/max | Bloom oracle | Range oracle | Range runs | Min/max / range | Bloom / range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in promising:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{result.query} {result.edge}",
                    str(result.intervals),
                    f"{result.minmax_survivors} ({percent(result.minmax_survivors, result.intervals)})",
                    f"{result.bloom_oracle_survivors} ({percent(result.bloom_oracle_survivors, result.intervals)})",
                    f"{result.range_oracle_survivors} ({percent(result.range_oracle_survivors, result.intervals)})",
                    str(result.range_survivor_runs),
                    ratio(result.minmax_survivors, result.range_oracle_survivors),
                    ratio(result.bloom_oracle_survivors, result.range_oracle_survivors),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful PRF-over-min/max case has multiple separated range-oracle runs and substantially fewer range survivors than min/max survivors. A useful PRF-over-Bloom case additionally has file intervals wider than Bloom's enumeration limit.",
            "",
            "The file proxy preserves the current database tuple order; it does not invent query-specific clustering. It predicts the file min/max intervals produced by a similarly ordered Parquet export, apart from compression-dependent file boundaries.",
            "",
            "## Existing measured profile scan rows",
            "",
            "These are total base-table rows reported as scanned in the already generated profiles. They are supporting execution evidence, not a substitute for the interval oracle: pipelines, limits, and plan changes can affect them.",
            "",
            "| Query | PRF targets | No prefilter | Bloom/SIP | SIP + min/max | PRF with fallback |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in profiles:
        lines.append(
            f"| {row['query']} | {row['prf_targets'] or '—'} | "
            f"{row['no_prefilter_scan_rows']:,} | {row['sip_scan_rows']:,} | "
            f"{row['sip_min_max_scan_rows']:,} | {row['prf_scan_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "The detailed interval audit covers direct selective dimension-to-fact edges. Complex filters derived from large fact-side subqueries (for example Q4 and Q18) remain visible in the measured profile table but are not reconstructed by this lightweight audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit TPC-H Skew join-filter prunability using storage min/max intervals."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-rows-per-file", type=int, default=1_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_rows_per_file <= 0:
        raise ValueError("--target-rows-per-file must be positive")
    db_path = args.database.resolve()
    profiles_dir = args.profiles_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    interval_cache: dict[tuple[str, str], list[Interval]] = {}
    results: list[EdgeResult] = []
    try:
        for edge in AUDIT_EDGES:
            key = (edge.probe_table, edge.probe_column)
            native_intervals = interval_cache.get(key)
            if native_intervals is None:
                native_intervals = extract_row_group_intervals(con, *key)
                interval_cache[key] = native_intervals
            results.append(audit_edge(con, edge, native_intervals, "row_group", None))
            file_intervals = bundle_intervals(native_intervals, args.target_rows_per_file)
            results.append(
                audit_edge(
                    con,
                    edge,
                    file_intervals,
                    "file_proxy",
                    args.target_rows_per_file,
                )
            )
    finally:
        con.close()

    profiles = profile_summary(profiles_dir)
    result_dicts = [asdict(result) for result in results]
    write_csv(output_dir / "edge_audit.csv", result_dicts)
    write_csv(output_dir / "profile_scan_summary.csv", profiles)
    payload = {
        "database": str(db_path),
        "profiles_dir": str(profiles_dir),
        "target_rows_per_file": args.target_rows_per_file,
        "bloom_enumeration_span": BLOOM_ENUMERATION_SPAN,
        "edges": result_dicts,
        "profiles": profiles,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(
        output_dir / "README.md",
        db_path,
        args.target_rows_per_file,
        results,
        profiles,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
