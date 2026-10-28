"""Resume a checkpoint on its original VM without replacing the pinned runner.

The original host module, container image, and benchmark code stay unchanged.
Only checkpoint restoration and the first S3 staging operation use local state.
"""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

try:
    from infra.aws.common import digest
except ModuleNotFoundError:
    from common import digest


def verify_local_checkpoint(_aws, _bucket, marker, destination, run_id):
    root = (Path(destination) / run_id).resolve()
    if marker.get("run_id") != run_id or marker.get("final") or not marker.get("files"):
        raise ValueError("local resume requires an unfinished checkpoint for this run")
    for relative, checksum in marker["files"].items():
        path = root / relative
        if (not path.resolve().is_relative_to(root) or path.is_symlink()
                or not path.is_file() or digest(path) != checksum):
            raise ValueError(f"local checkpoint differs: {relative}")


def reuse_staged_objects(scratch, run: Path, workload: str, catalog: str):
    """Authenticate the existing objects against both preparation and upload records."""
    scratch._name(workload)
    relative = PurePosixPath(catalog)
    if not relative.is_relative_to('/work') or '..' in relative.parts:
        raise ValueError("invalid local catalog path")
    local = (scratch.data_root / relative.relative_to('/work')).resolve()
    if not local.is_relative_to(scratch.data_root):
        raise ValueError("catalog escapes the data volume")
    files = json.loads(local.read_text())
    prepared = json.loads((local.parent / 'complete.json').read_text())
    report = json.loads((run / 's3' / (workload + '.json')).read_text())
    prefix = scratch.prefix + workload + '/'
    if report['base_url'] != f's3://{scratch.bucket}/{prefix.rstrip("/")}':
        raise ValueError("staged objects belong to another attempt")
    expected = {prefix + name: (table, name) for table, names in files.items() for name in names}
    uploaded = {item['key']: item for item in report['objects']}
    actual = {item['Key']: item['Size'] for item in scratch.objects(prefix)}
    if (set(expected) != set(uploaded) or len(uploaded) != len(report['objects'])
            or set(actual) != set(expected)):
        raise ValueError("staged object inventory differs")
    for key, (table, name) in expected.items():
        path = (local.parent / 'parquet' / name).resolve()
        if not path.is_relative_to(local.parent / 'parquet') or not path.is_file():
            raise ValueError("invalid prepared Parquet path")
        checksum = base64.b64encode(bytes.fromhex(prepared['data_files']['parquet/' + name])).decode()
        item = uploaded[key]
        if (item['table'] != table or item['sha256_base64'] != checksum
                or item['bytes'] != actual[key] or item['bytes'] != path.stat().st_size):
            raise ValueError("uploaded objects differ from preparation")

    def verify(key):
        value = scratch.aws.command('s3api', 'head-object', '--bucket', scratch.bucket,
                                    '--key', key, '--checksum-mode', 'ENABLED')
        if (value.get('ChecksumSHA256') != uploaded[key]['sha256_base64']
                or value.get('ContentLength') != uploaded[key]['bytes']):
            raise ValueError(f"S3 object differs from its upload record: {key}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, expected))
    # The saved staging report includes its old protocol token. Returning that
    # token would overwrite the new request's token in the host response.
    return {'base_url': report['base_url'], 'objects': report['objects'],
            'reused_existing_objects': True, 'verified_object_checksums': len(expected)}


def main():
    import host
    launch = json.loads(Path('/opt/join-pruning/launch.json').read_text())
    marker = launch.get('restore', {})
    if not launch.get('resume_local') or marker.get('attempt') != launch['attempt']:
        raise ValueError('local resume must retain the original attempt')
    host.restore_checkpoint = verify_local_checkpoint
    original_upload = host.ScratchData.upload
    reused = set()

    def upload(scratch, workload, catalog):
        report = Path('/srv/benchmark-results') / launch['run_id'] / 's3' / (workload + '.json')
        if workload not in reused and report.exists():
            value = reuse_staged_objects(scratch, report.parent.parent, workload, catalog)
            reused.add(workload)
            return value
        return original_upload(scratch, workload, catalog)
    host.ScratchData.upload = upload
    host.main()


if __name__ == '__main__':
    main()
