"""Validate physical Parquet ranges and DuckLake's registered pruning statistics."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmarking.resources import connect
from benchmarking.suite_io import write_json


def parquet_evidence(catalog_path: Path, *, ordered: bool = False) -> dict:
    root = catalog_path.parent / 'parquet'
    catalog = json.loads(catalog_path.read_text())
    evidence = {}
    with connect() as con:
        for table, files in catalog.items():
            for relative in files:
                path = root / relative
                rows = con.execute("""SELECT row_group_id, row_group_num_rows, path_in_schema,
                    stats_min_value, stats_max_value, stats_null_count
                    FROM parquet_metadata(?) ORDER BY row_group_id, column_id""", [str(path)]).fetchall()
                groups, columns = {}, {}
                for group, count, column, lower, upper, nulls in rows:
                    groups[group] = count
                    if column == 'fact_id' or column == 'd_key' or column.startswith('fk_'):
                        if lower is None or upper is None or nulls != 0:
                            raise ValueError(f"missing key bounds in {relative}/{column}")
                        columns.setdefault(column, []).append((int(lower), int(upper)))
                if not groups:
                    raise ValueError(f"empty Parquet file: {relative}")
                if table == 'fact' and ordered:
                    for column, ranges in columns.items():
                        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                            raise ValueError(f"unordered row-group bounds: {relative}/{column}")
                evidence[relative] = {"table": table, "rows": sum(groups.values()), "row_groups": len(groups),
                    "bytes": path.stat().st_size,
                    "columns": {name: {"min": min(a for a, _ in bounds), "max": max(b for _, b in bounds)}
                                for name, bounds in columns.items()}}
    facts = sorted((v for v in evidence.values() if v['table'] == 'fact'),
                   key=lambda v: v['columns']['fact_id']['min'])
    if ordered:
        for left, right in zip(facts, facts[1:]):
            for name, bounds in left['columns'].items():
                if bounds['max'] > right['columns'][name]['min']:
                    raise ValueError(f"fact files have overlapping ordered intervals: {name}")
    write_json(catalog_path.parent / 'parquet_statistics.json', evidence)
    return evidence


def compare_catalog(evidence: dict, rows: list[dict], base_url: str) -> dict:
    expected = {(base_url + '/' + file, column): (item, bounds)
                for file, item in evidence.items() for column, bounds in item['columns'].items()}
    seen = set()
    for row in rows:
        key = (row['path'], row['column_name'])
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate DuckLake file statistic: {key}")
        item, bounds = expected[key]
        if (row['table_name'] != item['table'] or row['record_count'] != item['rows']
                or row['file_size_bytes'] != item['bytes'] or row['min_value'] is None or row['max_value'] is None
                or int(row['min_value']) != bounds['min'] or int(row['max_value']) != bounds['max']):
            raise ValueError(f"DuckLake catalog differs from Parquet metadata: {key}")
        seen.add(key)
    if seen != expected.keys():
        raise ValueError(f"DuckLake catalog lacks {len(expected.keys() - seen)} key statistics")
    return {"validated_files": len(evidence), "validated_column_statistics": len(seen),
            "rows_by_table": {table: sum(v['rows'] for v in evidence.values() if v['table'] == table)
                              for table in {v['table'] for v in evidence.values()}}, "status": "complete"}


def validate_catalog(binary, workload, setup_sql) -> dict:
    sql = '\n'.join(setup_sql) + """
        SELECT t.table_name, d.path, d.record_count, d.file_size_bytes, c.column_name,
               s.min_value, s.max_value
        FROM jpb_meta.ducklake_data_file d
        JOIN jpb_meta.ducklake_table t ON t.table_id=d.table_id
        JOIN jpb_meta.ducklake_file_column_stats s ON s.data_file_id=d.data_file_id AND s.table_id=d.table_id
        JOIN jpb_meta.ducklake_column c ON c.table_id=d.table_id AND c.column_id=s.column_id
        WHERE d.end_snapshot IS NULL AND t.end_snapshot IS NULL AND c.end_snapshot IS NULL
          AND (c.column_name IN ('fact_id', 'd_key') OR starts_with(c.column_name, 'fk_'));
    """
    process = subprocess.run([str(binary), str(workload.db_path), '-json', '-c', sql],
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError('DuckLake catalog validation failed: ' + process.stderr)
    evidence = json.loads((workload.artifact_dir / 'parquet_statistics.json').read_text())
    # CREATE SECRET emits a separate JSON result before the catalog SELECT.
    # Decode the result stream, retaining the final query's rows.
    decoder = json.JSONDecoder()
    remaining = process.stdout.strip()
    rows = None
    while remaining:
        rows, offset = decoder.raw_decode(remaining)
        remaining = remaining[offset:].lstrip()
    if not isinstance(rows, list):
        raise ValueError('DuckLake catalog query produced no JSON result')
    report = compare_catalog(evidence, rows, workload.ducklake_s3_url)
    write_json(workload.artifact_dir / 'ducklake_catalog_validation.json', report)
    return report


def validate_remote_reads(results: Path, workload, report_path: Path) -> None:
    """A range baseline must exclude objects before reading their S3 footers."""
    evidence = json.loads((workload.artifact_dir / 'parquet_statistics.json').read_text())
    fact_files = sum(item['table'] == 'fact' for item in evidence.values())
    checks = []
    with connect(str(results), read_only=True) as con:
        for query, variant, repetition, parameters, total, fact, details in con.execute("""
            SELECT query_name, variant_label, repetition, query_parameters,
                   remote_bytes_read, fact_remote_bytes_read, remote_read_details FROM timing_runs
        """).fetchall():
            if total is None or fact is None or not 0 < fact <= total:
                raise ValueError(f"missing/nonpositive S3 data transfer: {query}/{variant}/{repetition}")
            objects = json.loads(details)
            accessed = {r['url'] for r in objects if '/fact/' in r['url'] and r['get_requests'] > 0}
            if (json.loads(parameters)['query_type'] == 'range' and fact_files >= 4
                    and variant in ('sip + min/max', 'prf with fallback') and len(accessed) >= fact_files):
                raise ValueError(f"range query failed catalog-level file pruning: {query}/{variant}")
            checks.append({'query': query, 'variant': variant, 'repetition': repetition,
                           'fact_objects_fetched': len(accessed), 'fact_objects_available': fact_files,
                           'fact_bytes': fact, 'total_bytes': total})
    if not checks:
        raise ValueError('no S3 measurements to validate')
    write_json(report_path, {'status': 'complete', 'measurements': checks})
