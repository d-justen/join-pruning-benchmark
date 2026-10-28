"""Checkpoint-boundary handoff for an explicitly authorized runtime extension.

Run on the original VM after extending its EC2 watchdog tag. The benchmark
container and old supervisor are replaced only after a verified checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

try:
    from infra.aws.common import digest, write_json
except ModuleNotFoundError:
    from common import digest, write_json


def extend_launch(original, seconds, expected_stop):
    if original['stop_at'] != expected_stop or not 0 < seconds <= 86400:
        raise ValueError('unexpected original deadline or invalid extension')
    return {**original, 'stop_at': original['stop_at'] + seconds,
            'measurement_deadline': original['measurement_deadline'] + seconds,
            'maximum_lifetime_seconds': original['maximum_lifetime_seconds'] + seconds,
            'resume_local': True}


def checkpoint_ready(sync, run):
    request, ack = sync / 'request.json', sync / 'ack.json'
    if not request.exists():
        return False
    token = json.loads(request.read_text())['token']
    if ack.exists() and json.loads(ack.read_text()).get('token') == token:
        return False
    states = [json.loads(p.read_text()) for p in (run / 'units').glob('*/status.json')]
    return bool(states) and all(s['status'] == 'complete' for s in states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-stop-at', required=True, type=float)
    parser.add_argument('--seconds', required=True, type=int)
    args = parser.parse_args()
    import host
    from local_resume import verify_local_checkpoint
    launch_path = Path('/opt/join-pruning/launch.json')
    original = json.loads(launch_path.read_text())
    if original['run_id'] != args.run_id:
        raise ValueError('unexpected run on this VM')
    updated = extend_launch(original, args.seconds, args.expected_stop_at)
    controller = host.Host(original)
    progress = controller.results / ('runtime-extension-' + original['attempt'] + '.json')
    def report(phase, **values):
        write_json(progress, {'phase': phase, 'at': time.time(), 'run_id': args.run_id,
                             'stop_at': updated['stop_at'],
                             'measurement_deadline': updated['measurement_deadline'], **values})
        print(phase, flush=True)
    def call(*command):
        return subprocess.run(command, check=True, capture_output=True, text=True)
    def interrupted(_signum, _frame):
        raise InterruptedError('extension interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    pid = int(call('systemctl', 'show', 'join-pruning.service', '--property=MainPID', '--value').stdout)
    if pid <= 1 or not Path('/opt/join-pruning/local_resume.py').is_file():
        raise ValueError('supervisor or local resume helper is missing')
    # Scheduling a new shutdown replaces the existing Linux shutdown timer.
    minutes = max(1, math.floor((updated['stop_at'] - time.time()) / 60))
    call('shutdown', '-h', '+' + str(minutes))
    paused = killed = started = False
    override = Path('/etc/systemd/system/join-pruning.service.d/30-local-resume.conf')
    try:
        # Only the host supervisor is stopped here. The query keeps executing.
        os.kill(pid, signal.SIGSTOP)
        report('waiting_for_checkpoint', supervisor_pid=pid)
        deadline = min(original['measurement_deadline'] - 120, time.time() + 3600)
        while not checkpoint_ready(controller.sync, controller.run):
            if time.time() >= deadline or not Path(f'/proc/{pid}').exists():
                raise TimeoutError('no complete query checkpoint before the handoff deadline')
            time.sleep(.25)
        call('docker', 'pause', controller.container)
        paused = True
        if not checkpoint_ready(controller.sync, controller.run):
            raise RuntimeError('checkpoint changed before container pause')
        report('verifying_data')
        inventory = {}
        for complete in Path('/mnt/benchmark/workloads').glob('*/complete.json'):
            saved = json.loads(complete.read_text())
            for relative, checksum in saved['data_files'].items():
                path = complete.parent / relative
                if not path.resolve().is_relative_to(complete.parent) or digest(path) != checksum:
                    raise ValueError(f'prepared data changed: {path}')
                stat = path.stat()
                inventory[str(path)] = {'sha256': checksum, 'bytes': stat.st_size,
                                        'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
        if not inventory:
            raise ValueError('no prepared data remain on this VM')
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        evidence = controller.run / 'cloud' / original['attempt'] / 'runtime-extensions' / stamp
        write_json(evidence / 'launch-before.json', original)
        write_json(evidence / 'prepared-cache.json', inventory)
        write_json(evidence / 'extension.json', {
            'seconds': args.seconds, 'old_stop_at': original['stop_at'],
            'new_stop_at': updated['stop_at'], 'old_measurement_deadline': original['measurement_deadline'],
            'new_measurement_deadline': updated['measurement_deadline'],
            'method': 'same VM; verified query checkpoint; original image; local resume',
            'old_supervisor_pid': pid, 'checkpoint_token': json.loads((controller.sync / 'request.json').read_text())['token']})
        for name in ('host.py', 'local_resume.py', 'extend_running.py'):
            shutil.copy2(Path('/opt/join-pruning') / name, evidence / name)
        latest = controller.aws.get_json(controller.bucket, controller.prefix + '/latest.json')
        if latest['attempt'] != original['attempt'] or latest.get('final'):
            raise ValueError('unexpected remote checkpoint')
        controller.snapshot_files = latest['files']
        controller.parts = latest.get('parts', [latest])
        controller.runtime = json.loads((controller.run / 'cloud' / original['attempt'] / 'runtime.json').read_text())
        report('uploading_checkpoint', prepared_files=len(inventory))
        controller.upload()
        marker = controller.aws.get_json(controller.bucket, controller.prefix + '/latest.json')
        verify_local_checkpoint(controller.aws, controller.bucket, marker, controller.results, args.run_id)
        updated['restore'] = marker
        updated['runtime_extension_evidence'] = str(evidence.relative_to(controller.run))
        write_json(launch_path, updated)
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/python3 /opt/join-pruning/local_resume.py\n')
        call('systemctl', 'daemon-reload')
        report('restarting_runner', prepared_files=len(inventory), checkpoint_sha256=marker['sha256'])
        # SIGKILL bypasses the old supervisor's unconditional VM-shutdown finally.
        # It is safe here: the container is paused at a verified, closed checkpoint.
        call('systemctl', 'kill', '--kill-whom=all', '--signal=SIGKILL', 'join-pruning.service')
        killed = True
        call('docker', 'rm', '--force', controller.container)
        paused = False
        call('systemctl', 'stop', 'join-pruning.service')
        for name in ('request.json', 'ack.json', 's3-request.json', 's3-response.json'):
            (controller.sync / name).unlink(missing_ok=True)
        call('systemctl', 'reset-failed', 'join-pruning.service')
        call('systemctl', 'start', 'join-pruning.service')
        started = True
        report('resumed', prepared_files=len(inventory), checkpoint_sha256=marker['sha256'],
               evidence=str(evidence), supervisor_pid=int(call('systemctl', 'show', 'join-pruning.service', '--property=MainPID', '--value').stdout))
    except BaseException as exc:
        report('failed', error=str(exc), old_supervisor_killed=killed, new_supervisor_started=started)
        raise
    finally:
        if not killed:
            if paused:
                call('docker', 'unpause', controller.container)
            if Path(f'/proc/{pid}').exists():
                os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
