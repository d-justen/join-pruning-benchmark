from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarking.cloud import Cloud, bootstrap, memory_bytes, validate_resources
from benchmarking.suite import DeadlineReached, check_deadline, execute
from infra.aws.common import AWS, extract_checkpoint, write_json
from infra.aws.template import template
from test_suite import SuiteIntegrationTests


class PublishPreflightTests(unittest.TestCase):
    def setUp(self):
        self.cloud = Cloud.__new__(Cloud)
        self.cloud.args = SimpleNamespace(image="join-pruning-suite:local")
        self.cloud.outputs = MagicMock()
        self.cloud.aws = MagicMock()

    def test_missing_image_explains_build_and_context_before_publication(self):
        failure = subprocess.CalledProcessError(1, ["docker", "image", "inspect"],
                                               stderr='Error response from daemon: {"message":"No such image: join-pruning-suite:local"}')
        with patch("benchmarking.cloud.subprocess.check_output", side_effect=failure):
            with self.assertRaises(SystemExit) as raised:
                self.cloud.publish()
        message = str(raised.exception)
        self.assertIn("docker context show", message)
        self.assertIn("./bench build --target paper --image join-pruning-suite:local", message)
        self.cloud.outputs.assert_not_called()
        self.assertEqual(self.cloud.aws.mock_calls, [])

    def test_daemon_failure_preserves_diagnostic_without_suggesting_rebuild(self):
        failure = subprocess.CalledProcessError(1, ["docker", "image", "inspect"],
                                               stderr="Cannot connect to the Docker daemon")
        with patch("benchmarking.cloud.subprocess.check_output", side_effect=failure):
            with self.assertRaises(SystemExit) as raised:
                self.cloud.publish()
        self.assertIn("Cannot connect to the Docker daemon", str(raised.exception))
        self.assertNotIn("./bench build", str(raised.exception))
        self.cloud.outputs.assert_not_called()

    def test_missing_docker_explains_prerequisite(self):
        with patch("benchmarking.cloud.subprocess.check_output", side_effect=FileNotFoundError("docker")):
            with self.assertRaisesRegex(SystemExit, "Docker CLI not found"):
                self.cloud.publish()
        self.cloud.outputs.assert_not_called()


class CloudTests(unittest.TestCase):
    def test_small_host_rejects_full_config(self):
        root = Path(__file__).resolve().parents[1]
        instance = {"ProcessorInfo": {"SupportedArchitectures": ["arm64"]}, "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384}, "InstanceStorageInfo": {"TotalSizeInGB": 237}}
        validate_resources(json.loads((root / "configs/smoke.json").read_text()), instance)
        with self.assertRaisesRegex(ValueError, "CPUs"):
            validate_resources(json.loads((root / "configs/graviton-full.json").read_text()), instance)
        self.assertEqual(memory_bytes("6GiB"), 6 * 1024**3)

    def test_bootstrap_arms_shutdown_before_install_and_pins_bundle(self):
        spec = {"region": "eu-central-1", "bucket": "test-bucket", "stop_at": 2000000000,
                "run_id": "test", "attempt": "attempt", "config": {"large": "x" * 50000},
                "aws_cli": {"version": "2.38.3", "sha256": "0" * 64}}
        script = bootstrap(spec, "bundles/worker.tar.gz", "a" * 64)
        self.assertLess(len(script.encode()), 16384)
        self.assertLess(script.index("shutdown -h +"), script.index("apt-get update"))
        self.assertIn("sha256sum -c", script)
        self.assertIn("Restart=no", script)

    def test_watchdog_is_independent_and_scoped(self):
        value = template()
        self.assertEqual(value["Resources"]["WatchdogRule"]["Properties"]["ScheduleExpression"], "rate(1 minute)")
        role = value["Resources"]["WatchdogRole"]["Properties"]
        statements = role["Policies"][0]["PolicyDocument"]["Statement"]
        stop = next(s for s in statements if "ec2:StopInstances" in s["Action"])
        self.assertIn("ec2:ResourceTag/BenchStack", stop["Condition"]["StringEquals"])
        self.assertEqual(value["Resources"]["Bucket"]["DeletionPolicy"], "Retain")

    def test_archive_rejects_links_traversal_and_wrong_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, kind in (("run/../../escape", tarfile.REGTYPE), ("other/file", tarfile.REGTYPE),
                               ("run/link", tarfile.SYMTYPE)):
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.linkname = "/tmp"
                    tar.addfile(info)
                with self.assertRaises(ValueError):
                    extract_checkpoint(archive, root / "restore", "run")

    def test_bad_download_removed_and_upload_checksum_required(self):
        aws = AWS("eu-central-1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file"
            path.write_text("data")
            with patch.object(aws, "command", return_value={}):
                with self.assertRaisesRegex(ValueError, "confirm"):
                    aws.put("bucket", "key", path)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    aws.get("bucket", "key", path, "0" * 64)
                self.assertFalse(path.exists())

    def test_deadline(self):
        with patch.dict(os.environ, {"BENCH_DEADLINE_EPOCH": str(time.time() - 1)}):
            with self.assertRaises(DeadlineReached):
                check_deadline()

    def test_resume_rejects_pending_reservations_and_newly_visible_instances(self):
        from unittest.mock import MagicMock
        instance = {"ProcessorInfo": {"SupportedArchitectures": ["arm64"]}, "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384}, "InstanceStorageInfo": {"TotalSizeInGB": 237}}
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/smoke.json").read_text())
        for known in ([], [{"InstanceId": "i-other", "State": {"Name": "running"},
                            "Tags": [{"Key": "BenchAttempt", "Value": "other"}]}]):
            with self.subTest(known=known), tempfile.TemporaryDirectory() as tmp:
                cloud = Cloud.__new__(Cloud)
                cloud.args = SimpleNamespace(run_id="test", max_runtime_seconds=600, instance_type=None)
                cloud.settings = {"max_runtime_seconds": 21600}
                cloud.local = Path(tmp)
                cloud.outputs = MagicMock(return_value={"Bucket": "bucket", "AvailabilityZone": "zone",
                                                        "WatchdogRule": "rule", "Watchdog": "watchdog"})
                cloud.instances = MagicMock(side_effect=[[], known])
                cloud.doctor = MagicMock(return_value={"quota_ok": True, "zones": ["zone"]})
                cloud.aws = MagicMock()
                cloud.aws.get_json.side_effect = [
                    {"config": config, "published": {}, "instance_type": "m9gd.xlarge"},
                    RuntimeError("NoSuchKey"),
                    ({"attempt": "other", "at": time.time()}, {"ETag": "previous"}),
                ]
                cloud.aws.command.side_effect = [{"InstanceTypes": [instance]}, {"State": "ENABLED"}, {}]
                with self.assertRaisesRegex(RuntimeError, "reserved|active instance"):
                    cloud.launch(resume=True)
                cloud.aws.put_json.assert_not_called()
                self.assertFalse(any(c.args[:2] == ("ec2", "run-instances") for c in cloud.aws.command.call_args_list))

    def test_cleanup_preserves_orphan_volume_without_final_delivery(self):
        from unittest.mock import MagicMock
        cloud = Cloud.__new__(Cloud)
        cloud.args = SimpleNamespace(run_id="test", delete_volumes=True)
        cloud.stack = "stack"
        cloud.instances = MagicMock(return_value=[])
        cloud.outputs = MagicMock(return_value={"Bucket": "bucket"})
        cloud.aws = MagicMock()
        cloud.aws.command.return_value = {"Volumes": [{"VolumeId": "vol-results", "State": "available",
                                                       "Tags": [{"Key": "BenchAttempt", "Value": "attempt"}]}]}
        cloud.aws.get_json.return_value = {"final": False}
        with self.assertRaisesRegex(ValueError, "final delivery"):
            cloud.cleanup()
        self.assertFalse(any(c.args[:2] == ("ec2", "delete-volume") for c in cloud.aws.command.call_args_list))

    def test_case_resume_retries_only_corrupt_or_missing_outputs(self):
        from benchmarking.case_checkpoints import run_case
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "emit.py"
            script.write_text("import sys\nfrom pathlib import Path\np=Path(sys.argv[1].split('=',1)[1])\np.write_text('a,b\\n1,2\\n')\n")
            output = root / "case.csv"
            command = [sys.executable, str(script)]
            run_case(command, output, {"config": 1})
            with patch("benchmarking.case_checkpoints.subprocess.run", side_effect=AssertionError("reran completed case")):
                run_case(command, output, {"config": 1})
            output.write_text("corrupt")
            run_case(command, output, {"config": 1})
            self.assertEqual(output.read_text(), "a,b\n1,2\n")

    def test_watchdog_stops_only_overdue_instances(self):
        import runpy
        from unittest.mock import MagicMock
        client = MagicMock()
        instances = [{"InstanceId": name, "State": {"Name": "running"}, "Tags": [{"Key": "BenchStopAt", "Value": str(deadline)}]}
                     for name, deadline in (("expired", 1), ("future", time.time() + 1000))]
        client.get_paginator.return_value.paginate.return_value = [{"Reservations": [{"Instances": instances}]}]
        with patch.dict("sys.modules", {"boto3": SimpleNamespace(client=lambda _: client)}), patch.dict(os.environ, {"STACK": "test"}):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/aws/watchdog.py"))
            module["handler"]({}, None)
        client.stop_instances.assert_called_once_with(InstanceIds=["expired"], Force=True)

    def test_jpb_case_execution_preserves_order_and_deterministic_results(self):
        from test_jpb_microbenchmarks import write_config, BINARY
        from benchmarking.jpb_microbenchmarks import run_jpb_microbenchmarks
        from benchmarking.duckdb_build import ResolvedSystem
        from benchmarking.microbenchmarks import read_csv
        if not BINARY.exists():
            self.skipTest("native JPB binary required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root)
            system = ResolvedSystem("binary", "duckdb-paper", config.systems[0].binary_path,
                                    "fixture-host", None, None, None, build_dir=BINARY.parent)
            with patch("benchmarking.jpb_microbenchmarks.resolve_binary", return_value=BINARY):
                whole = run_jpb_microbenchmarks(config, system, root / "whole")
                split = run_jpb_microbenchmarks(config, system, root / "split", checkpoint_cases=True)
                import subprocess
                original_run = subprocess.run
                def reject_case(command, *args, **kwargs):
                    if any(str(arg).startswith("--case-index=") for arg in command):
                        raise AssertionError("reran case")
                    return original_run(command, *args, **kwargs)
                with patch("benchmarking.case_checkpoints.subprocess.run", side_effect=reject_case):
                    run_jpb_microbenchmarks(config, system, root / "split", checkpoint_cases=True)
            fields = ("query", "filter", "repetition", "build_checksum", "probe_checksum", "build_seed", "true_positives", "false_positives")
            self.assertEqual([[r[k] for k in fields] for r in read_csv(whole / "measurements.csv")],
                             [[r[k] for k in fields] for r in read_csv(split / "measurements.csv")])

    def test_failed_upload_never_publishes_delivery_marker(self):
        import runpy
        from unittest.mock import MagicMock
        from infra.aws import common
        with patch.dict("sys.modules", {"common": common}):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/aws/host.py"))
        with tempfile.TemporaryDirectory() as tmp:
            spec = {"region": "eu-central-1", "bucket": "test", "run_id": "run", "attempt": "a",
                    "stop_at": time.time() + 1000, "measurement_deadline": time.time() + 900}
            host = module["Host"](spec)
            host.results = Path(tmp)
            host.run = Path(tmp) / "run"
            host.run.mkdir()
            (host.run / "result.txt").write_text("valid results")
            host.aws = MagicMock()
            host.aws.put.side_effect = RuntimeError("network unavailable")
            with self.assertRaises(RuntimeError):
                host.upload(final=True)
            host.aws.put_json.assert_not_called()
            self.assertTrue((host.run / "result.txt").exists())

    def test_disk_discovery_excludes_root_via_partition_tree(self):
        import runpy
        from unittest.mock import MagicMock
        from infra.aws import common
        with patch.dict("sys.modules", {"common": common}):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/aws/host.py"))
        devices = [
            {"path": "/dev/root", "type": "disk", "model": "Amazon Elastic Block Store", "mountpoints": [None],
             "children": [{"path": "/dev/root1", "type": "part", "mountpoints": ["/"]}]},
            {"path": "/dev/results", "type": "disk", "model": "Amazon Elastic Block Store", "mountpoints": [None]},
            {"path": "/dev/scratch", "type": "disk", "model": "Amazon EC2 NVMe Instance Storage", "mountpoints": [None]},
        ]
        def output(command):
            if command[0] == "lsblk":
                self.assertIn("--tree", command)
                return json.dumps({"blockdevices": devices}).encode()
            return b'{"signatures": []}'
        call = MagicMock()
        globals_ = module["mount_storage"].__globals__
        with patch.dict(globals_, {"call": call, "Path": MagicMock()}), \
             patch("os.path.ismount", return_value=False), \
             patch("subprocess.check_output", side_effect=output), \
             patch("subprocess.run", return_value=SimpleNamespace(returncode=2, stdout="")):
            module["mount_storage"]()
        formatted = [c.args[-1] for c in call.call_args_list if c.args[0] == "mkfs.ext4"]
        self.assertEqual(formatted, ["/dev/scratch", "/dev/results"])

    def test_incremental_restore_deletions_and_final_checksums(self):
        import runpy
        import shutil
        from infra.aws import common
        with patch.dict("sys.modules", {"common": common}):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/aws/host.py"))
        class Store:
            def __init__(self):
                self.objects = {}
            def put(self, bucket, key, path):
                self.objects[key] = Path(path).read_bytes()
            def put_json(self, bucket, key, value):
                self.objects[key] = json.dumps(value).encode()
            def get(self, bucket, key, path, checksum):
                Path(path).write_bytes(self.objects[key])
                if common.digest(path) != checksum:
                    raise ValueError("bad checksum")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"region": "eu-central-1", "bucket": "test", "run_id": "run", "attempt": "a",
                    "stop_at": time.time() + 1000, "measurement_deadline": time.time() + 900}
            host = module["Host"](spec)
            host.results, host.run = root / "source", root / "source/run"
            host.run.mkdir(parents=True)
            host.aws = Store()
            (host.run / "old.txt").write_text("old")
            host.upload()
            (host.run / "old.txt").unlink()
            (host.run / "new.txt").write_text("new")
            host.upload()
            marker = json.loads(host.aws.objects["runs/run/latest.json"])
            self.assertEqual(len(marker["parts"]), 2)
            common.restore_checkpoint(host.aws, "test", marker, root / "restored", "run")
            self.assertFalse((root / "restored/run/old.txt").exists())
            self.assertEqual((root / "restored/run/new.txt").read_text(), "new")
            host.upload(final=True, exit_code=0)
            marker = json.loads(host.aws.objects["runs/run/latest.json"])
            self.assertNotIn("parts", marker)
            common.restore_checkpoint(host.aws, "test", marker, root / "final", "run")
            for line in (root / "final/run/SHA256SUMS").read_text().splitlines():
                checksum, name = line.split("  ", 1)
                self.assertEqual(common.digest(root / "final/run" / name), checksum)


class CloudSuiteTests(SuiteIntegrationTests):
    def test_deadline_exports_partial_and_restores_on_new_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.configuration()
            config, run = root / "config.json", root / "run"
            write_json(config, raw)
            with patch.dict(os.environ, {"BENCH_DEADLINE_EPOCH": str(time.time() - 1)}):
                self.assertEqual(execute(config, "all", run, root / "cache", False), 1)
            self.assertTrue((root / "run.tar.gz").exists())
            self.assertEqual(json.loads((run / "run.json").read_text())["interruption"], "deadline")
            self.assertEqual(execute(config, "all", run, root / "fresh-cache", True), 0)

    def test_query_checkpoints_survive_later_variants_and_resume(self):
        import duckdb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.configuration()
            raw["suite"]["checkpoint_queries"] = True
            raw["workloads"][0]["params"]["query_types"].append({"kind": "shoulder", "probe_layouts": ["uniform"]})
            raw["systems"][0]["variants"].append({"label": "second", "setup_sql": []})
            config, run, cache = root / "config.json", root / "run", root / "cache"
            write_json(config, raw)
            self.assertEqual(execute(config, "all", run, cache, False), 0)
            states = [json.loads(p.read_text()) for p in (run / "units").glob("*/status.json")]
            self.assertEqual(len([s for s in states if s["action"] == "measure"]), 4)
            self.assertEqual(execute(config, "all", run, cache, True), 0)
            with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 4)


if __name__ == "__main__":
    unittest.main()
