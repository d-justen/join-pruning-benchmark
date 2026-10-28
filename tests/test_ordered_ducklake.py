from __future__ import annotations

import json
import base64
import hashlib
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

import duckdb

from benchmarking import load_experiment_config
from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.duckdb_runner import benchmark_workload, remote_bytes_snapshot_sql
from benchmarking.ducklake_validation import compare_catalog, validate_catalog, validate_remote_reads
from benchmarking.experiment_config import applies_to_workload
from benchmarking.http_file_server import build_handler
from benchmarking.jpb_microbenchmarks import ARTIFACT_SUBDIR, resolve_jpb_cases, run_jpb_microbenchmarks
from benchmarking.manifests import DUCKLAKE_S3_BASE_URL_PLACEHOLDER
from benchmarking.s3_data import data_scope
from benchmarking.suite_jobs import job_configs
from benchmarking.workloads.cross_product import prepare_cross_product_workload
from benchmarking.suite_coverage import expected_parameters
from infra.aws.scratch import ScratchData
from paper.figure_specs import figures_for_config
from paper.figures.build_all import build_figures
from paper.figures.data import MetricRow, ProbeScanRow, PlanMetricRow
from paper.figures.jpb_microbenchmark import build_jpb_artifacts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/graviton-ordered-ducklake.json'
NATIVE = ROOT / 'duckdb/build/paper-native-release/duckdb'


def small_config(root):
    raw = json.loads(CONFIG.read_text())
    raw['workloads'] = [raw['workloads'][1]]
    raw['systems'] = [raw['systems'][0]]
    raw['systems'][0]['variants'] = [v for v in raw['systems'][0]['variants'] if 'workload_names' not in v]
    raw.pop('jpb_microbenchmarks')
    fixed = raw['workloads'][0]['params']['fixed']
    fixed.update(fact_rows=65536, dimension_cardinality=10000, selectivity=.01)
    raw['workloads'][0]['params']['sweep']['values'] = [10000]
    raw['workloads'][0]['params']['storage'].update(parquet_row_group_size=2048, parquet_row_groups_per_file=2)
    path = root / 'config.json'
    path.write_text(json.dumps(raw))
    return load_experiment_config(path)


class OrderedConfigTests(unittest.TestCase):
    @unittest.skipUnless(NATIVE.with_name('tools').joinpath('join_pruning_benchmark').is_file(), 'requires native JPB binary')
    def test_seasonal_variants_run_in_cpp_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = json.loads(CONFIG.read_text())
            raw['workloads'] = raw['workloads'][:1]
            raw['workloads'][0]['params']['fixed'].update(fact_rows=65536, dimension_cardinality=10000)
            raw['workloads'][0]['params']['sweep']['values'] = [10000]
            raw['jpb_microbenchmarks'].update(repetitions=1, warmups=0, required_filters=['prf_uncompressed', 'prf'])
            (root / 'config.json').write_text(json.dumps(raw))
            config = load_experiment_config(root / 'config.json')
            system = ResolvedSystem('binary', 'duckdb-paper', NATIVE, 'test', None, None, None)
            run_jpb_microbenchmarks(config, system, root)
            artifacts = build_jpb_artifacts(root / ARTIFACT_SUBDIR / 'measurements.csv', root / 'paper',
                                             root / 'paper/figures', [])
            self.assertEqual({r['pattern'] for r in artifacts['series']},
                             {'range', 'shoulder', 'seasonal_8', 'seasonal_16', 'seasonal_32', 'scattered'})

    def test_queries_variants_microbenchmarks_and_figures(self):
        config = load_experiment_config(CONFIG)
        cases = resolve_jpb_cases(config)
        self.assertEqual(len(cases), 30)
        self.assertEqual({c['pattern'] for c in cases}, {'range', 'shoulder', 'seasonal_8', 'seasonal_16', 'seasonal_32', 'scattered'})
        self.assertEqual({c['dimension_cardinality'] for c in cases}, {100000, 1000000, 10000000, 100000000, 1000000000})
        self.assertEqual({c['build_keys'] for c in cases if c['dimension_cardinality'] == 1000000000}, {10000000})
        self.assertEqual({c['layout'] for c in cases}, {'ordered'})
        figures = {f.slug: f for f in figures_for_config(config)}
        self.assertTrue(figures['join_pruning_ordered'].facet_dimensions)
        self.assertEqual(len(figures['join_pruning_ordered'].queries_by_workload['join-pruning-workload']), 30)
        lake = figures['join_pruning_ordered_ducklake']
        self.assertEqual(lake.metric, 'runtime_and_bytes_read')
        self.assertEqual(len(lake.queries_by_workload['join-pruning-workload-ducklake']), 6)
        variants = [v.label for s in config.systems if applies_to_workload(s, 'join-pruning-workload-ducklake')
                    for v in s.variants if applies_to_workload(v, 'join-pruning-workload-ducklake')]
        self.assertEqual(variants, ['no prefilter', 'sip', 'sip + min/max', 'prf with fallback'])
        raw = json.loads(CONFIG.read_text())
        raw.pop('jpb_microbenchmarks')
        jobs = job_configs(raw)
        self.assertEqual(len(jobs[1]['systems']), 1)
        self.assertEqual([v['label'] for v in jobs[1]['systems'][0]['variants']], variants)

    def test_payload_export_order_and_catalog_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config(root)
            manifest = prepare_cross_product_workload(config.workloads[0], root / 'data')
            evidence = json.loads((manifest.artifact_dir / 'parquet_statistics.json').read_text())
            facts = [f for f in evidence.values() if f['table'] == 'fact']
            self.assertEqual(len(facts), 16)
            self.assertEqual({f['row_groups'] for f in facts}, {2})
            parquet = str(manifest.artifact_dir / 'parquet/fact/*.parquet')
            with duckdb.connect() as con:
                count, distinct, lengths, different = con.execute('''SELECT count(*), count(DISTINCT payload_s1),
                    count(DISTINCT length(payload_s1)), bool_and(payload_s1 != payload_s2) FROM read_parquet(?)''', [parquet]).fetchone()
                self.assertEqual((count, distinct, lengths, different), (65536, 65536, 1, True))
                payload = con.execute('SELECT payload_s1 FROM read_parquet(?) WHERE fact_id=1', [parquet]).fetchone()[0]
                self.assertEqual(len(payload), 32)
                self.assertEqual(payload, base64.b64encode(hashlib.sha256(b'fact:7:1:0:1').digest()).decode()[:32])
            queries = list(manifest.queries_dir.glob('*.sql'))
            self.assertEqual(len(queries), 6)
            for query in queries:
                sql = query.read_text()
                self.assertIn('hash(f.payload_s1, f.payload_s2)', sql)
                self.assertNotIn('f.payload_s3', sql)
            bootstrap = manifest.ducklake_bootstrap_sql_path.read_text()
            self.assertIn('ducklake_add_data_files', bootstrap)
            self.assertIn(DUCKLAKE_S3_BASE_URL_PLACEHOLDER, bootstrap)
            self.assertIn("METADATA_CATALOG 'jpb_meta'", bootstrap)
            base = 's3://bucket/scratch/run/attempt/workload'
            rows = [dict(path=base + '/' + name, table_name=f['table'], record_count=f['rows'],
                         file_size_bytes=f['bytes'], column_name=column,
                         min_value=str(bounds['min']), max_value=str(bounds['max']))
                    for name, f in evidence.items() for column, bounds in f['columns'].items()]
            self.assertEqual(compare_catalog(evidence, rows, base)['validated_files'], len(evidence))
            for invalid in (rows[:-1], rows + [rows[0]], [{**rows[0], 'min_value': None}, *rows[1:]]):
                with self.assertRaises(ValueError):
                    compare_catalog(evidence, invalid, base)

    def test_byte_snapshot_excludes_unrelated_data_and_keeps_fact_separate(self):
        with duckdb.connect() as con:
            con.execute('''CREATE MACRO httpfs_transfer_stats() AS TABLE SELECT * FROM (VALUES
                ('https://bucket/scratch/run/fact/a.parquet', 100, 2),
                ('https://bucket/scratch/run/dim/a.parquet', 50, 1),
                ('https://bucket/scratch/other/fact/a.parquet', 999, 9))
                t(url, bytes_received, get_requests)''')
            con.execute(remote_bytes_snapshot_sql(SimpleNamespace(ducklake_s3_url='s3://bucket/scratch/run')))
            value = con.execute("SELECT getvariable('benchmark_io')").fetchone()[0]
            self.assertEqual((value['total_bytes'], value['fact_bytes'], value['requests']), (150, 100, 3))
            self.assertEqual(len(json.loads(value['details'])), 2)


class FakeS3:
    def __init__(self):
        self.files = {}
        self.fail_after = None
        self.versioning = {}
        self.deletions = []

    def command(self, service, operation, *args, payload=None):
        if operation == 'get-bucket-versioning':
            return self.versioning
        if operation == 'list-objects-v2':
            prefix = args[args.index('--prefix') + 1]
            keys = sorted(k for k in self.files if k.startswith(prefix))
            start = int(args[args.index('--continuation-token') + 1]) if '--continuation-token' in args else 0
            return {'Contents': [{'Key': k} for k in keys[start:start + 1000]],
                    'IsTruncated': len(keys) > start + 1000, 'NextContinuationToken': str(start + 1000)}
        if operation == 'delete-objects':
            for obj in payload['Delete']['Objects']:
                self.deletions.append(obj['Key'])
                self.files.pop(obj['Key'], None)
            return {}
        raise AssertionError(operation)

    def put(self, bucket, key, path):
        if self.fail_after == 0:
            raise RuntimeError('upload failed')
        if self.fail_after is not None:
            self.fail_after -= 1
        self.files[key] = path.read_bytes()
        return {'ChecksumSHA256': 'test-checksum'}


class OrderedFigureTests(unittest.TestCase):
    def test_complete_native_and_bytes_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = json.loads(CONFIG.read_text())
            raw.pop('jpb_microbenchmarks')
            (root / 'config.json').write_text(json.dumps(raw))
            (root / 'results.duckdb').touch()
            rows, scans, plans = [], [], []
            for workload in raw['workloads']:
                name = workload['name']
                variants = [v['label'] for s in raw['systems'] if applies_to_workload(s, name)
                            for v in s['variants'] if applies_to_workload(v, name)]
                for query, params in expected_parameters(workload).items():
                    params = {**params, 'dimension_cardinality': params['sweep_value'],
                              'cluster_count': next((n for n in (8, 16, 32) if f'seasonal_c{n:05d}' in query), 0)}
                    for index, variant in enumerate(variants):
                        rows.append(MetricRow(name, query, variant, 10 + index, 50000, 60000,
                                              query_parameters=params, fact_remote_bytes_read=(index + 1) * 1048576,
                                              remote_bytes_read=(index + 1) * 1048576 + 1000))
                        scans.append(ProbeScanRow(name, query, variant, 'fact', 'fact', 50000, 10000, '',
                                                  total_files_read=5, table_file_count=16))
                        plans.append(PlanMetricRow(name, query, variant, 60000, 10000, 1))
            with patch('paper.figures.build_all.load_system_names', return_value=('duckdb-paper', 'rpt-plus')), \
                 patch('paper.figures.build_all.load_metric_rows', return_value=rows), \
                 patch('paper.figures.build_all.load_probe_scan_rows', return_value=scans), \
                 patch('paper.figures.build_all.load_plan_metric_rows', return_value=plans):
                output = build_figures(root, None)
            native = json.loads((output / 'join_pruning_ordered_series.json').read_text())
            lake = json.loads((output / 'join_pruning_ordered_ducklake_series.json').read_text())
            self.assertEqual(len(native), 30)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['verification']['join_pruning_ordered']['dimension_panels'], 5)
            self.assertEqual(len(lake), 6)
            self.assertEqual({len(e['series']) for e in lake}, {4})
            for entry in lake:
                self.assertEqual([s['fact_remote_gib'] for s in entry['series']], [n / 1024 for n in (1, 2, 3, 4)])
            for slug in ('join_pruning_ordered', 'join_pruning_ordered_ducklake'):
                self.assertGreater((output / 'figures' / (slug + '.pdf')).stat().st_size, 1000)


class ScratchTests(unittest.TestCase):
    def test_delete_error_is_not_reported_as_success(self):
        aws = FakeS3()
        scratch = ScratchData(aws, 'bucket', 'run', 'attempt')
        aws.files[scratch.prefix + 'w/fact/a'] = b'data'
        command = aws.command
        def fail_delete(service, operation, *args, **kwargs):
            return {'Errors': [{'Code': 'AccessDenied'}]} if operation == 'delete-objects' else command(service, operation, *args, **kwargs)
        with patch.object(aws, 'command', side_effect=fail_delete):
            with self.assertRaisesRegex(RuntimeError, 'S3 scratch deletion failed'):
                scratch.delete()
        self.assertTrue(aws.files)

    def test_paginated_deletion_preserves_other_attempts(self):
        aws = FakeS3()
        scratch = ScratchData(aws, 'bucket', 'run', 'attempt')
        aws.files = {scratch.prefix + 'w/fact/' + str(i): b'x' for i in range(1001)}
        aws.files['scratch/run/other/w/fact/0'] = b'keep'
        result = scratch.delete()
        self.assertEqual(result['deleted_objects'], 1001)
        self.assertTrue(result['verified_empty'])
        self.assertEqual(list(aws.files), ['scratch/run/other/w/fact/0'])

    def test_upload_failure_cleans_partial_data_and_rejects_versioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'parquet/fact').mkdir(parents=True)
            for name in ('a', 'b'):
                (root / f'parquet/fact/{name}').write_bytes(b'parquet')
            (root / 'catalog.json').write_text(json.dumps({'fact': ['fact/a', 'fact/b']}))
            aws = FakeS3()
            scratch = ScratchData(aws, 'bucket', 'run', 'attempt', data_root=root)
            aws.fail_after = 1
            with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                scratch.upload('w', '/work/catalog.json')
            self.assertEqual(aws.files, {})
            self.assertEqual(aws.deletions, ['scratch/run/attempt/w/fact/a'])
            aws.versioning = {'Status': 'Enabled'}
            with self.assertRaisesRegex(ValueError, 'versioning'):
                scratch.upload('w', '/work/catalog.json')
            with self.assertRaises(ValueError):
                scratch.upload('../other', '/work/catalog.json')

    def test_context_cleans_after_query_and_staging_failure(self):
        manifest = SimpleNamespace(ducklake_data_source='s3')
        for failure in ('query', 'upload'):
            with patch('benchmarking.s3_data.stage', side_effect=RuntimeError('upload') if failure == 'upload' else None), \
                 patch('benchmarking.s3_data.cleanup') as cleanup:
                with self.assertRaises(RuntimeError):
                    with data_scope(manifest, Path('/run')):
                        raise RuntimeError('query')
                cleanup.assert_called_once()


@unittest.skipUnless(os.environ.get('JPB_DUCKLAKE_TEST_BINARY'), 'requires pinned DuckLake/httpfs build')
class DuckLakeTransferIntegrationTests(unittest.TestCase):
    def test_catalog_pruning_and_actual_bytes_with_both_http_backends(self):
        binary = Path(os.environ['JPB_DUCKLAKE_TEST_BINARY'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config(root)
            manifest = prepare_cross_product_workload(config.workloads[0], root / 'data')
            handler = build_handler(manifest.artifact_dir / 'parquet', 0)
            handler.protocol_version = 'HTTP/1.1'
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                bootstrap = manifest.ducklake_bootstrap_sql_path.read_text().replace(DUCKLAKE_S3_BASE_URL_PLACEHOLDER, base)
                setup = manifest.setup_sql_paths[0].read_text().replace(DUCKLAKE_S3_BASE_URL_PLACEHOLDER, base)
                # S3 setup emits a Success result before the catalog JSON rows.
                # These dummy credentials never apply to the HTTP test server.
                setup = "CREATE OR REPLACE SECRET jpb_test (TYPE s3, KEY_ID 'test', SECRET 'test');\n" + setup
                def run(sql):
                    result = subprocess.run([str(binary), str(manifest.db_path), '-json', '-c', sql],
                                            capture_output=True, text=True, restore_signals=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return result.stdout
                run(bootstrap)
                validate_catalog(binary, replace(manifest, ducklake_s3_url=base), (setup,))
                query = next(p for p in manifest.queries_dir.glob('*.sql') if 'range' in p.name).read_text()
                fact_count = len(json.loads(manifest.parquet_catalog_path.read_text())['fact'])
                for backend in ('httplib', 'curl'):
                    answers, reads = [], []
                    for variant in config.systems[0].variants:
                        urlopen(base + '/__reset_stats').close()
                        output = run(setup + f"SET threads=2; SET httpfs_client_implementation='{backend}';" + '\n'.join(variant.setup_sql)
                                     + 'SELECT httpfs_reset_transfer_stats();' + query
                                     + 'SELECT url, bytes_received, get_requests FROM httpfs_transfer_stats();')
                        decoder = json.JSONDecoder()
                        values = []
                        while output.strip():
                            value, offset = decoder.raw_decode(output.lstrip())
                            values.append(value)
                            output = output.lstrip()[offset:]
                        answers.append({k: values[-2][0][k] for k in ('row_count', 'payload_checksum')})
                        transfers = values[-1]
                        with urlopen(base + '/__stats') as response:
                            stats = json.load(response)
                        self.assertEqual(sum(int(v['bytes_received']) for v in transfers), stats['bytes_sent'])
                        # Includes attempts cancelled before the server sends a response.
                        self.assertGreaterEqual(sum(int(v['get_requests']) for v in transfers), stats['get_requests'])
                        fact = [v for v in transfers if '/fact/' in v['url']]
                        self.assertGreater(sum(int(v['bytes_received']) for v in fact), 0)
                        reads.append(len(fact))
                    self.assertTrue(all(answer == answers[0] for answer in answers))
                    self.assertEqual(reads[0], fact_count)
                    self.assertLess(reads[2], fact_count)
                    self.assertLess(reads[3], fact_count)
                # Exercise the real timing SQL, warmup/reset boundaries, profiles,
                # stored byte fields and figure reader for all six predicates.
                remote_manifest = replace(manifest, ducklake_s3_url=base)
                system = ResolvedSystem('binary', 'duckdb-paper', binary, 'test', None, None, None)
                options = replace(config.benchmark, repetitions=2, threads=2)
                with patch('benchmarking.duckdb_runner.remote_s3_setup', return_value='LOAD httpfs;'):
                    benchmark_workload(config.experiment_name, remote_manifest, system, config.systems[0].variants,
                                       options, root / 'results.duckdb', root / 'sql', root / 'profiles')
                validate_remote_reads(root / 'results.duckdb', remote_manifest, root / 'reads.json')
                output = build_figures(root, None)
                series = json.loads((output / 'join_pruning_ordered_ducklake_series.json').read_text())
                self.assertEqual(len(series), 6)
                self.assertEqual({len(e['series']) for e in series}, {4})
                # Missing bounds must retain files even when PRF would exclude them.
                run(setup + "UPDATE jpb_meta.ducklake_file_column_stats SET min_value=NULL, max_value=NULL;")
                output = run(setup + '\n'.join(config.systems[0].variants[-1].setup_sql)
                             + 'SELECT httpfs_reset_transfer_stats();' + query
                             + "SELECT count(*) AS files FROM httpfs_transfer_stats() WHERE contains(url, '/fact/');")
                self.assertIn(f'"files":{fact_count}', output)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
