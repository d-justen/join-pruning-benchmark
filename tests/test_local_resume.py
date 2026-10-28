import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from infra.aws.common import digest, write_json
from infra.aws.extend_running import checkpoint_ready, extend_launch
from infra.aws.local_resume import reuse_staged_objects, verify_local_checkpoint
from infra.aws.scratch import ScratchData


class LocalResumeTests(unittest.TestCase):
    def test_extension_retains_workload_and_finalization_reserve(self):
        original = {'stop_at': 10000, 'measurement_deadline': 9220,
                    'maximum_lifetime_seconds': 6000, 'config': {'fixed': 1}, 'attempt': 'same'}
        updated = extend_launch(original, 21600, 10000)
        self.assertEqual(updated['stop_at'] - updated['measurement_deadline'], 780)
        self.assertEqual(updated['config'], original['config'])
        self.assertEqual(updated['attempt'], 'same')
        self.assertEqual(original['stop_at'], 10000)
        for seconds, old in ((0, 10000), (21600, 11000), (86401, 10000)):
            with self.assertRaises(ValueError):
                extend_launch(original, seconds, old)

    def test_handoff_requires_unacknowledged_checkpoint_and_no_active_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sync, run = root / 'sync', root / 'run'
            write_json(sync / 'request.json', {'token': 'next'})
            write_json(sync / 'ack.json', {'token': 'previous'})
            write_json(run / 'units/query/status.json', {'status': 'running'})
            self.assertFalse(checkpoint_ready(sync, run))
            write_json(run / 'units/query/status.json', {'status': 'complete'})
            self.assertTrue(checkpoint_ready(sync, run))
            write_json(sync / 'ack.json', {'token': 'next'})
            self.assertFalse(checkpoint_ready(sync, run))

    def test_local_restore_preserves_files_and_rejects_corruption_and_wrong_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            file = destination / 'run' / 'result.json'
            write_json(file, {'result': 42})
            marker = {'run_id': 'run', 'final': False, 'files': {'result.json': digest(file)}}
            stat = file.stat()
            verify_local_checkpoint(None, None, marker, destination, 'run')
            self.assertEqual(file.stat().st_mtime_ns, stat.st_mtime_ns)
            for changed in ({**marker, 'run_id': 'other'}, {**marker, 'final': True},
                            {**marker, 'files': {'../outside': '0' * 64}}):
                with self.assertRaises(ValueError):
                    verify_local_checkpoint(None, None, changed, destination, 'run')
            file.write_text('corrupt')
            with self.assertRaises(ValueError):
                verify_local_checkpoint(None, None, marker, destination, 'run')

    def test_s3_reuse_verifies_objects_without_upload_or_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            aws = MagicMock()
            scratch = ScratchData(aws, 'bucket', 'run', 'attempt', data_root=root / 'data')
            directory = root / 'data' / 'dataset'
            data = directory / 'parquet/fact/0.parquet'
            data.parent.mkdir(parents=True)
            data.write_bytes(b'prepared parquet bytes')
            checksum = base64.b64encode(bytes.fromhex(digest(data))).decode()
            write_json(directory / 'complete.json', {'data_files': {'parquet/fact/0.parquet': digest(data)}})
            write_json(directory / 'parquet_catalog.json', {'fact': ['fact/0.parquet']})
            key = 'scratch/run/attempt/workload/fact/0.parquet'
            report = {'token': 'previous-upload-request',
                      'base_url': 's3://bucket/scratch/run/attempt/workload',
                      'objects': [{'key': key, 'table': 'fact', 'bytes': data.stat().st_size,
                                   'sha256_base64': checksum}]}
            run = root / 'run'
            write_json(run / 's3/workload.json', report)
            def respond(service, operation, *args):
                if operation == 'list-objects-v2':
                    return {'Contents': [{'Key': key, 'Size': data.stat().st_size}]}
                self.assertEqual(operation, 'head-object')
                return {'ChecksumSHA256': checksum, 'ContentLength': data.stat().st_size}
            aws.command.side_effect = respond
            result = reuse_staged_objects(scratch, run, 'workload', '/work/dataset/parquet_catalog.json')
            self.assertEqual(result['verified_object_checksums'], 1)
            self.assertTrue(result['reused_existing_objects'])
            self.assertNotIn('token', result)
            self.assertEqual({'token': 'current-request', **result}['token'], 'current-request')
            aws.put.assert_not_called()
            for invalid in ('checksum', 'inventory'):
                def bad(service, operation, *args):
                    result = respond(service, operation, *args)
                    if invalid == 'checksum' and operation == 'head-object':
                        result['ChecksumSHA256'] = 'wrong'
                    if invalid == 'inventory' and operation == 'list-objects-v2':
                        result['Contents'] = []
                    return result
                aws.command.side_effect = bad
                with self.assertRaises(ValueError):
                    reuse_staged_objects(scratch, run, 'workload', '/work/dataset/parquet_catalog.json')


if __name__ == '__main__':
    unittest.main()
