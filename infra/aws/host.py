"""Unattended EC2 execution. Data are disposable; all run artifacts live on EBS."""
from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

from common import AWS, digest, restore_checkpoint, write_json
try:
    from infra.aws.scratch import ScratchData
    from infra.aws.checkpoint_resume import rebind_legacy_ducklake_units
except ModuleNotFoundError:  # The EC2 worker bundle contains standalone modules.
    from scratch import ScratchData
    from checkpoint_resume import rebind_legacy_ducklake_units


def call(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def mount_storage():
    # --tree is essential when selecting PATH rather than NAME: otherwise
    # lsblk emits a flat list and a mounted root partition's parent looks unused.
    devices = json.loads(subprocess.check_output(["lsblk", "--json", "--tree", "--output", "PATH,TYPE,MODEL,MOUNTPOINTS"]))["blockdevices"]
    def mounted(device):
        return any(device.get("mountpoints") or []) or any(mounted(c) for c in device.get("children", []))
    scratch, durable = [], []
    for device in devices:
        if device["type"] != "disk" or mounted(device):
            continue
        model = (device.get("model") or "").replace(" ", "")
        if model == "AmazonEC2NVMeInstanceStorage":
            scratch.append(device["path"])
        elif model == "AmazonElasticBlockStore":
            durable.append(device["path"])
    for candidates, target in ((scratch, "/mnt/benchmark"), (durable, "/srv/benchmark-results")):
        if os.path.ismount(target):
            continue
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one unmounted disk for {target}: {candidates}")
        device = candidates[0]
        probe = subprocess.run(["blkid", "-o", "value", "-s", "TYPE", device], capture_output=True, text=True)
        if probe.returncode == 2:
            # An unexpected partition table/signature is never overwritten.
            signatures = json.loads(subprocess.check_output(["wipefs", "--json", device]))
            if signatures.get("signatures"):
                raise RuntimeError(f"refusing to format disk with existing signatures: {device}")
            call("mkfs.ext4", "-q", device)
        elif probe.returncode or probe.stdout.strip() != "ext4":
            raise RuntimeError(f"unsupported filesystem on {device}: {probe.stdout}")
        Path(target).mkdir(parents=True, exist_ok=True)
        call("mount", device, target)


class Host:
    def __init__(self, launch):
        self.spec = launch
        self.aws = AWS(launch["region"])
        self.bucket = launch["bucket"]
        self.prefix = "runs/" + launch["run_id"]
        self.results = Path("/srv/benchmark-results")
        self.run = self.results / launch["run_id"]
        self.sync = Path("/srv/benchmark-sync")
        self.deadline = launch["stop_at"]
        self.measure_deadline = launch["measurement_deadline"]
        self.container = "join-pruning-" + launch["attempt"]
        self.last_token = None
        restored = launch.get("restore") or {}
        self.snapshot_files = restored.get("files", {})
        self.parts = restored.get("parts", [restored] if restored else [])
        self.runtime = {}
        self.scratch = (ScratchData(self.aws, launch["scratch_bucket"], launch["run_id"], launch["attempt"],
                                    deadline=self.measure_deadline) if launch.get("scratch_bucket") else None)
        self.last_s3_token = None

    def status(self, phase, **fields):
        value = {"phase": phase, "attempt": self.spec["attempt"], "run_id": self.spec["run_id"],
                 "at": time.time(), **fields}
        write_json(self.results / ("host-" + self.spec["attempt"] + ".json"), value)
        self.aws.put_json(self.bucket, self.prefix + "/status.json", value)

    def cleanup_scratch(self):
        if self.scratch is None:
            return True
        try:
            report = self.scratch.delete()
            write_json(self.run / "s3" / "host-cleanup.json", report)
            return True
        except Exception as exc:
            write_json(self.run / "s3" / "host-cleanup.json", {"error": str(exc), "verified_empty": False})
            print("S3 cleanup failed; scratch lifecycle expiration remains active:", exc, flush=True)
            return False

    def upload(self, final=False, exit_code=None):
        if not self.run.exists() and not final:
            return
        self.run.mkdir(parents=True, exist_ok=True)
        # Called only while the suite is paused between units or has exited.
        evidence = self.run / "cloud" / self.spec["attempt"]
        evidence.mkdir(parents=True, exist_ok=True)
        restored = self.spec.get("restore")
        write_json(evidence / "launch.json", {**self.spec, "restore":
                   {k: v for k, v in restored.items() if k not in ("files", "parts")} if restored else None})
        write_json(evidence / "runtime.json", self.runtime)
        for name in ("cloud-init-output.log", "join-pruning.log"):
            source = Path("/var/log") / name
            if source.exists():
                # Logs are evidence, not part of measurement checkpoint validity.
                (evidence / name).write_bytes(source.read_bytes())
        files = {str(p.relative_to(self.run)): digest(p) for p in sorted(self.run.rglob("*"))
                 if p.is_file() and not p.is_symlink() and p.name not in (".lock", "SHA256SUMS")}
        if final:
            sums = self.run / "SHA256SUMS"
            sums.write_text("".join(f"{checksum}  {name}\n" for name, checksum in files.items()))
            files["SHA256SUMS"] = digest(sums)
        archive = self.results / (self.spec["attempt"] + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for name, checksum in files.items():
                if final or self.snapshot_files.get(name) != checksum:
                    tar.add(self.run / name, arcname=self.run.name + "/" + name, recursive=False)
        # A deadline can interrupt the pause while the host is capturing files.
        # Never advertise a snapshot whose manifest differs from its bytes.
        with tarfile.open(archive) as tar:
            for member in tar:
                name = str(Path(member.name).relative_to(self.run.name))
                with tar.extractfile(member) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != files[name]:
                        raise RuntimeError("snapshot changed during capture; retry at the next checkpoint")
        key = self.prefix + "/checkpoints/" + self.spec["attempt"] + "/" + str(time.time_ns()) + ".tar.gz"
        self.aws.put(self.bucket, key, archive)
        marker = {"key": key, "sha256": digest(archive), "bytes": archive.stat().st_size,
                  "run_id": self.spec["run_id"], "attempt": self.spec["attempt"], "final": final,
                  "exit_code": exit_code, "at": time.time()}
        marker["files"] = files
        if not final:
            marker["parts"] = [*self.parts, {"key": key, "sha256": marker["sha256"]}]
        if (self.run / "run.json").exists():
            marker["experiment_status"] = json.loads((self.run / "run.json").read_text()).get("status")
        else:
            marker["experiment_status"] = "bootstrap_failed"
        self.aws.put_json(self.bucket, self.prefix + "/latest.json", marker)
        self.snapshot_files = files
        self.parts = marker.get("parts", [{"key": key, "sha256": marker["sha256"]}])
        if final:
            self.aws.put_json(self.bucket, self.prefix + "/attempts/" + self.spec["attempt"] + "/delivered.json", marker)
        archive.unlink()

    def docker(self, stage, *, resume=False):
        config = self.spec["config"]
        command = ["docker", "run", "--rm", "--init", "--name", self.container,
                   "--memory", config["suite"]["container_memory"],
                   "--memory-swap", config["suite"]["container_memory"],
                   "--cpus", str(config["suite"]["container_cpus"]),
                   "--env", "BENCH_UID=1000", "--env", "BENCH_GID=1000",
                   "--env", "BENCH_IMAGE_METADATA=" + json.dumps(self.image_metadata),
                   "--mount", "type=bind,src=/mnt/benchmark,dst=/work",
                   "--mount", "type=bind,src=/srv/benchmark-results,dst=/results",
                   "--mount", "type=bind,src=/opt/join-pruning/config.json,dst=/input/config.json,readonly"]
        if stage == "all":
            command += ["--env", "BENCH_DEADLINE_EPOCH=" + str(self.measure_deadline),
                        "--env", "BENCH_SYNC_DIR=/sync", "--mount", f"type=bind,src={self.sync},dst=/sync"]
        command += ["--env", "AWS_REGION=" + self.spec["region"]]
        command += [self.spec["image"], stage, "--run", "/results/" + self.spec["run_id"], "--cache", "/work"]
        if stage == "all":
            command += ["--config", "/input/config.json"]
        if resume:
            command += ["--resume"]
        return command

    def execute(self):
        mount_storage()
        self.sync.mkdir(parents=True, exist_ok=True)
        for directory in (self.results, self.sync, Path("/mnt/benchmark")):
            os.chown(directory, 1000, 1000)
        self.status("starting")
        self.runtime = {
            "docker": json.loads(subprocess.check_output(["docker", "version", "--format", "{{json .}}"])),
            "aws_cli": subprocess.check_output(["aws", "--version"], text=True).strip(),
            "os_release": Path("/etc/os-release").read_text(),
            "disks": json.loads(subprocess.check_output(["lsblk", "--json", "--tree", "--bytes", "--output", "NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS"])),
        }
        write_json("/opt/join-pruning/config.json", self.spec["config"])
        if self.spec.get("restore") and self.spec["restore"].get("experiment_status") != "bootstrap_failed":
            marker = self.spec["restore"]
            restore_checkpoint(self.aws, self.bucket, marker, self.results, self.spec["run_id"])
            if self.scratch is not None:
                repaired = rebind_legacy_ducklake_units(self.run, self.spec["scratch_bucket"], self.spec["attempt"])
                print(f"Rebound {repaired} completed DuckLake checkpoints to the new S3 attempt", flush=True)
            call("chown", "-R", "1000:1000", self.run)
        token = self.aws.command("ecr", "get-authorization-token")["authorizationData"][0]
        import base64
        password = base64.b64decode(token["authorizationToken"]).decode().split(":", 1)[1]
        call("docker", "login", "--username", "AWS", "--password-stdin", token["proxyEndpoint"], input=password, text=True)
        call("docker", "pull", "--platform", "linux/arm64", self.spec["image"], timeout=max(1, self.measure_deadline - time.time()))
        info = json.loads(subprocess.check_output(["docker", "image", "inspect", self.spec["image"]]))[0]
        accepted_ids = {self.spec["image_id"], self.spec["image"].split("@")[1]}
        if info["Architecture"] != "arm64" or info["Id"] not in accepted_ids:
            raise ValueError("pulled image does not match the published ARM64 image identity")
        self.image_metadata = {"id": self.spec["image_id"], "docker_id": info["Id"], "repo_digests": info.get("RepoDigests", []), "architecture": info["Architecture"], "platform": info["Os"]}
        self.status("running")
        process = subprocess.Popen(self.docker("all", resume=bool(self.spec.get("restore"))))
        cleanup_failed = False
        try:
            while process.poll() is None:
                if time.time() >= self.measure_deadline + 30:
                    call("docker", "stop", "--time", "15", self.container, timeout=30)
                    process.wait(timeout=30)
                    break
                s3_request = self.sync / "s3-request.json"
                if s3_request.exists():
                    request = json.loads(s3_request.read_text())
                    if request["token"] != self.last_s3_token:
                        response = {"token": request["token"]}
                        try:
                            if self.scratch is None:
                                raise RuntimeError("scratch bucket is missing; deploy the updated infrastructure")
                            if request["operation"] == "upload":
                                response.update(self.scratch.upload(request["workload"], request["catalog"]))
                            elif request["operation"] == "delete":
                                response.update(self.scratch.delete(request["workload"]))
                            else:
                                raise ValueError("unknown S3 operation")
                        except Exception as exc:
                            response["error"] = str(exc)
                        write_json(self.sync / "s3-response.json", response)
                        self.last_s3_token = request["token"]
                request = self.sync / "request.json"
                if request.exists():
                    token_ = json.loads(request.read_text())["token"]
                    if token_ != self.last_token:
                        error = None
                        try:
                            self.upload()
                        except Exception as exc:
                            error = str(exc)
                            print("Checkpoint upload failed:", error, flush=True)
                        write_json(self.sync / "ack.json", {"token": token_, "error": error})
                        self.last_token = token_
                time.sleep(.5)
        finally:
            try:
                if process.poll() is None:
                    subprocess.run(["docker", "kill", self.container], timeout=30)
                    process.wait(timeout=30)
            finally:
                cleanup_failed = not self.cleanup_scratch()
        code = process.returncode or int(cleanup_failed)
        self.status("finalizing", exit_code=code)
        # Export is retried even if analysis or signal handling failed.
        for stage in ("analyze", "export"):
            remaining = self.deadline - time.time() - 90
            if remaining < 30 or not (self.run / "config.json").exists():
                break
            try:
                call(*self.docker(stage), timeout=min(remaining, 300))
            except Exception as exc:
                print(f"Final {stage}: {exc}", flush=True)
                subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=30)
        if cleanup_failed and (self.run / "run.json").exists():
            state = json.loads((self.run / "run.json").read_text())
            state.update(status="partial", failures=[*state.get("failures", []), "S3 scratch cleanup failed"])
            write_json(self.run / "run.json", state)
        self.upload(final=True, exit_code=code)
        self.status("delivered", exit_code=code)


def main():
    host = Host(json.loads(Path("/opt/join-pruning/launch.json").read_text()))
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError("EC2 host termination requested")
    signal.signal(signal.SIGTERM, terminate)
    try:
        host.execute()
    except BaseException as exc:
        traceback.print_exc()
        try:
            host.upload(final=True, exit_code=1)
            host.status("failed", error=str(exc))
        except Exception:
            traceback.print_exc()
            write_json("/var/log/join-pruning-failure.json", {"error": str(exc), "delivery": "failed", "at": time.time()})
    finally:
        # EBS volumes are deliberately retained even when S3 is unavailable.
        subprocess.run(["shutdown", "-h", "now"], timeout=15)


if __name__ == "__main__":
    main()
