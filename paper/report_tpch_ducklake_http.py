from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import duckdb


VARIANTS = (
    ("no prefilter", "no_prefilter"),
    ("sip", "sip"),
    ("sip + min/max", "sip_min_max"),
    ("prf uncompressed", "prf_uncompressed"),
    ("prf with fallback", "prf_with_fallback"),
)
EXPERIMENT_NAME = "paper-tpch-skew-ducklake-http-10ms"
REPETITIONS = 3


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_seconds(milliseconds: float) -> str:
    return f"{milliseconds / 1000:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the DuckLake/HTTP TPC-H Skew results.")
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    results_db = root / "results.duckdb"

    con = duckdb.connect(str(results_db), read_only=True)
    try:
        timing_rows = con.execute(
            """
            SELECT query_name, variant_label, median(elapsed_ns) / 1000000.0 AS median_ms
            FROM timing_runs
            WHERE experiment_name = ?
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [EXPERIMENT_NAME],
        ).fetchall()
        profile_rows = con.execute(
            """
            SELECT query_name, variant_label, total_rows_scanned, total_bytes_read,
                   table_scan_file_count
            FROM profile_runs
            WHERE experiment_name = ?
            ORDER BY 1, 2
            """,
            [EXPERIMENT_NAME],
        ).fetchall()
        timing_count = con.execute(
            "SELECT count(*) FROM timing_runs WHERE experiment_name = ?",
            [EXPERIMENT_NAME],
        ).fetchone()[0]
        profile_count = con.execute(
            "SELECT count(*) FROM profile_runs WHERE experiment_name = ?",
            [EXPERIMENT_NAME],
        ).fetchone()[0]
        q2_profiles = con.execute(
            """
            SELECT variant_label, profile_path
            FROM profile_runs
            WHERE experiment_name = ? AND query_name = 'q02.sql'
            ORDER BY variant_label
            """,
            [EXPERIMENT_NAME],
        ).fetchall()
    finally:
        con.close()

    expected_queries = [f"q{number:02d}.sql" for number in range(1, 23)]
    expected_timing_count = len(expected_queries) * len(VARIANTS) * REPETITIONS
    expected_profile_count = len(expected_queries) * len(VARIANTS)
    timing_by_query: dict[str, dict[str, float]] = {query: {} for query in expected_queries}
    for query_name, variant_label, median_ms in timing_rows:
        timing_by_query[query_name][variant_label] = float(median_ms)
    if timing_count != expected_timing_count or profile_count != expected_profile_count:
        raise ValueError(
            f"incomplete experiment: expected {expected_timing_count} timings/"
            f"{expected_profile_count} profiles, got "
            f"{timing_count}/{profile_count}"
        )
    for query_name, values in timing_by_query.items():
        missing = [label for label, _ in VARIANTS if label not in values]
        if missing:
            raise ValueError(f"{query_name} is missing variants: {', '.join(missing)}")

    result_rows: list[dict[str, object]] = []
    for query_name in expected_queries:
        values = timing_by_query[query_name]
        baseline = values["no prefilter"]
        sip = values["sip"]
        min_max = values["sip + min/max"]
        prf_uncompressed = values["prf uncompressed"]
        prf = values["prf with fallback"]
        result_rows.append(
            {
                "query": query_name,
                "no_prefilter_ms": round(baseline, 3),
                "sip_ms": round(sip, 3),
                "sip_min_max_ms": round(min_max, 3),
                "prf_uncompressed_ms": round(prf_uncompressed, 3),
                "prf_with_fallback_ms": round(prf, 3),
                "prf_uncompressed_speedup_vs_no_prefilter": round(
                    baseline / prf_uncompressed, 4
                ),
                "prf_speedup_vs_no_prefilter": round(baseline / prf, 4),
                "prf_speedup_vs_sip": round(sip / prf, 4),
                "prf_speedup_vs_sip_min_max": round(min_max / prf, 4),
                "prf_compression_speedup": round(prf_uncompressed / prf, 4),
            }
        )
    write_csv(root / "results.csv", list(result_rows[0]), result_rows)

    profile_export = [
        {
            "query": row[0],
            "variant": row[1],
            "total_rows_scanned": row[2],
            "total_bytes_read": row[3],
            "table_scan_file_count": row[4],
        }
        for row in profile_rows
    ]
    write_csv(root / "profile_metrics.csv", list(profile_export[0]), profile_export)

    parquet_catalog = json.loads(
        (
            root
            / "workloads"
            / "tpch-skew-ducklake-http-10ms"
            / "parquet_catalog.json"
        ).read_text(encoding="utf-8")
    )
    partsupp_file_count = len(parquet_catalog["partsupp"])
    q2_file_rows: list[dict[str, object]] = []
    for variant_label, profile_path in q2_profiles:
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        stack = list(profile.get("children", []))
        files_read = 0
        scan_count = 0
        while stack:
            node = stack.pop()
            stack.extend(node.get("children", []))
            extra_info = node.get("extra_info", {})
            if node.get("operator_type") != "TABLE_SCAN" or extra_info.get("Table") != "partsupp":
                continue
            scan_count += 1
            files_read += int(extra_info["Total Files Read"])
        possible_file_opens = scan_count * partsupp_file_count
        q2_file_rows.append(
            {
                "variant": variant_label,
                "partsupp_scans": scan_count,
                "files_opened": files_read,
                "possible_file_opens": possible_file_opens,
                "files_opened_fraction": round(files_read / possible_file_opens, 6),
            }
        )
    variant_order = {label: index for index, (label, _) in enumerate(VARIANTS)}
    q2_file_rows.sort(key=lambda row: variant_order[str(row["variant"])])
    write_csv(root / "q02_partsupp_files.csv", list(q2_file_rows[0]), q2_file_rows)

    total_ms = {
        label: sum(timing_by_query[query][label] for query in expected_queries)
        for label, _ in VARIANTS
    }
    prf_label = "prf with fallback"
    aggregate_rows: list[dict[str, object]] = []
    for label, slug in VARIANTS:
        speedups = [
            timing_by_query[query]["no prefilter"] / timing_by_query[query][label]
            for query in expected_queries
        ]
        aggregate_rows.append(
            {
                "variant": label,
                "total_of_query_medians_ms": round(total_ms[label], 3),
                "total_speedup_vs_no_prefilter": round(total_ms["no prefilter"] / total_ms[label], 4),
                "geomean_speedup_vs_no_prefilter": round(geometric_mean(speedups), 4),
                "queries_faster_than_no_prefilter": sum(value > 1 for value in speedups),
            }
        )
    write_csv(root / "aggregate_results.csv", list(aggregate_rows[0]), aggregate_rows)

    http_stats_dir = (
        root
        / "profiles"
        / "tpch-skew-ducklake-http-10ms"
        / "duckdb-paper-ducklake-http"
        / "http_stats"
    )
    http_rows: list[dict[str, object]] = []
    for label, slug in VARIANTS:
        stats = json.loads((http_stats_dir / f"{slug}.json").read_text(encoding="utf-8"))
        http_rows.append(
            {
                "variant": label,
                "requests": stats["requests"],
                "range_get_requests": stats["range_get_requests"],
                "bytes_sent": stats["bytes_sent"],
                "gigabytes_sent_decimal": round(stats["bytes_sent"] / 1_000_000_000, 3),
                "unique_paths": len(stats.get("paths", {})),
            }
        )
    write_csv(root / "http_stats.csv", list(http_rows[0]), http_rows)

    prf_vs = {
        comparison: [
            timing_by_query[query][comparison] / timing_by_query[query][prf_label]
            for query in expected_queries
        ]
        for comparison in (
            "no prefilter",
            "sip",
            "sip + min/max",
            "prf uncompressed",
        )
    }
    notable_queries = ("q02.sql", "q04.sql", "q09.sql", "q16.sql", "q17.sql", "q19.sql")
    lines = [
        "# TPC-H Skew on DuckLake over HTTP (+10 ms/request)",
        "",
        f"The experiment completed all 22 queries for all {len(VARIANTS)} variants with "
        f"{REPETITIONS} measured repetitions per query. "
        "Each profile was collected in a separate fifth execution after one warm-up and the three timings.",
        "",
        f"- Timing rows: {timing_count} "
        f"(22 queries × {len(VARIANTS)} variants × {REPETITIONS} repetitions)",
        f"- Profile rows: {profile_count} (22 queries × {len(VARIANTS)} variants)",
        "- Parquet prefetching: disabled for every variant (the enabled path hit DuckDB's HTTP read-ahead negative-offset bug)",
        "- HTTP latency: fixed 10 ms before every GET/HEAD response from the local threaded endpoint",
        "",
        "## Aggregate runtime",
        "",
        "| Variant | Sum of query medians (s) | Speedup vs no prefilter | Geomean speedup vs no prefilter | Faster queries |",
        "|---|---:|---:|---:|---:|",
    ]
    for aggregate in aggregate_rows:
        lines.append(
            f"| {aggregate['variant']} | {format_seconds(float(aggregate['total_of_query_medians_ms']))} | "
            f"{float(aggregate['total_speedup_vs_no_prefilter']):.2f}× | "
            f"{float(aggregate['geomean_speedup_vs_no_prefilter']):.2f}× | "
            f"{aggregate['queries_faster_than_no_prefilter']}/22 |"
        )
    lines.extend(
        [
            "",
            "PRF comparison over the 22 individual queries:",
            "",
            f"- vs no prefilter: {geometric_mean(prf_vs['no prefilter']):.2f}× geometric-mean speedup; "
            f"faster on {sum(value > 1 for value in prf_vs['no prefilter'])}/22 queries",
            f"- vs Bloom (SIP): {geometric_mean(prf_vs['sip']):.2f}× geometric-mean speedup; "
            f"faster on {sum(value > 1 for value in prf_vs['sip'])}/22 queries",
            f"- vs Bloom + min/max: {geometric_mean(prf_vs['sip + min/max']):.2f}× geometric-mean speedup; "
            f"faster on {sum(value > 1 for value in prf_vs['sip + min/max'])}/22 queries",
            f"- vs uncompressed PRF: {geometric_mean(prf_vs['prf uncompressed']):.2f}× "
            f"geometric-mean speedup; faster on "
            f"{sum(value > 1 for value in prf_vs['prf uncompressed'])}/22 queries",
            "",
            "## Selected queries",
            "",
            "| Query | No prefilter (ms) | Bloom (ms) | Bloom + min/max (ms) | "
            "Uncompressed PRF (ms) | PRF (ms) | PRF vs no prefilter | PRF vs min/max | PRF vs uncompressed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_query = {row["query"]: row for row in result_rows}
    for query in notable_queries:
        row = by_query[query]
        lines.append(
            f"| {query.removesuffix('.sql').upper()} | {float(row['no_prefilter_ms']):.1f} | "
            f"{float(row['sip_ms']):.1f} | {float(row['sip_min_max_ms']):.1f} | "
            f"{float(row['prf_uncompressed_ms']):.1f} | "
            f"{float(row['prf_with_fallback_ms']):.1f} | "
            f"{float(row['prf_speedup_vs_no_prefilter']):.2f}× | "
            f"{float(row['prf_speedup_vs_sip_min_max']):.2f}× | "
            f"{float(row['prf_compression_speedup']):.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Q2 `partsupp` file access",
            "",
            "Q2 scans the 105-file `partsupp` table twice, so there are 210 possible file opens.",
            "",
            "| Variant | Files opened | Fraction opened |",
            "|---|---:|---:|",
        ]
    )
    for row in q2_file_rows:
        lines.append(
            f"| {row['variant']} | {row['files_opened']}/{row['possible_file_opens']} | "
            f"{100 * float(row['files_opened_fraction']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## HTTP access totals",
            "",
            "These counters include five executions per query: warm-up, three timed repetitions, and profiling.",
            "",
            "| Variant | Range requests | Data sent (GB) | Unique Parquet files |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in http_rows:
        lines.append(
            f"| {row['variant']} | {int(row['range_get_requests']):,} | "
            f"{float(row['gigabytes_sent_decimal']):.1f} | {row['unique_paths']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Compressed/fallback PRF totals {format_seconds(total_ms['prf with fallback'])} s across the query "
            f"medians, compared with {format_seconds(total_ms['prf uncompressed'])} s for uncompressed PRF "
            f"({total_ms['prf uncompressed'] / total_ms['prf with fallback']:.2f}× compressed-over-uncompressed "
            "speedup). Disabling compression also disables post-build analysis, the run index, and Bloom/min-max "
            "fallback registration, so this comparison measures that complete policy rather than compression cost alone.",
            "",
            "The endpoint is a controlled latency emulator, not an S3 model: it has no bandwidth cap, connection setup, "
            "TLS, retries, shared-service variance, or object-store request concurrency limits.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main()
