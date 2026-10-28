"""Standard-library AWS CLI transport shared by the launcher and EC2 host."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AWS:
    def __init__(self, region, profile=None, credentials=None):
        self.region, self.profile = region, profile
        self.credentials = credentials or {}

    def assume(self, role):
        result = self.command("sts", "assume-role", "--role-arn", role, "--role-session-name", "join-pruning-cli")
        creds = result["Credentials"]
        return AWS(self.region, credentials={"AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
                   "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"], "AWS_SESSION_TOKEN": creds["SessionToken"]})

    def command(self, service, operation, *args, payload=None, timeout=180):
        command = ["aws", "--region", self.region, "--no-cli-pager", "--output", "json",
                   "--cli-connect-timeout", "10", "--cli-read-timeout", "60"]
        if self.profile:
            command += ["--profile", self.profile]
        command += [service, operation, *map(str, args)]
        with tempfile.TemporaryDirectory() as directory:
            if payload is not None:
                path = Path(directory) / "request.json"
                write_json(path, payload)
                command += ["--cli-input-json", "file://" + str(path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                    env={**os.environ, **self.credentials, "AWS_MAX_ATTEMPTS": "2", "AWS_RETRY_MODE": "standard"})
        if result.returncode:
            raise RuntimeError(f"AWS {service} {operation}: {result.stderr.strip()}")
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def put(self, bucket, key, path, **conditions):
        path = Path(path)
        if path.stat().st_size >= 5 * 1024**3:
            raise ValueError("checkpoint exceeds single-upload limit; retain EBS and use smaller checkpoints")
        checksum = digest(path)
        args = ["--bucket", bucket, "--key", key, "--body", path,
                "--checksum-algorithm", "SHA256", "--checksum-sha256",
                base64.b64encode(bytes.fromhex(checksum)).decode()]
        for key_, value in conditions.items():
            args += ["--" + key_.replace("_", "-"), value]
        result = self.command("s3api", "put-object", *args, timeout=240)
        if result.get("ChecksumSHA256") != base64.b64encode(bytes.fromhex(checksum)).decode():
            raise ValueError("S3 did not confirm the supplied SHA-256")
        return result

    def put_json(self, bucket, key, value, **conditions):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json(path, value)
            return self.put(bucket, key, path, **conditions)

    def get(self, bucket, key, path, checksum=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self.command("s3api", "get-object", "--bucket", bucket, "--key", key, path, timeout=240)
        if checksum and digest(path) != checksum:
            path.unlink()
            raise ValueError(f"checksum mismatch: {key}")
        return response

    def get_json(self, bucket, key, *, metadata=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            response = self.get(bucket, key, path)
            value = json.loads(path.read_text())
            return (value, response) if metadata else value


def extract_checkpoint(archive, destination, run_id):
    """Reject links and unexpected roots before restoring any files."""
    import tarfile
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != run_id or not (member.isfile() or member.isdir())):
                raise ValueError("unsafe checkpoint archive member: " + member.name)
        tar.extractall(destination, filter="data")


def restore_checkpoint(aws, bucket, marker, destination, run_id):
    """Restore verified incremental snapshots, or a compact final archive."""
    destination = Path(destination)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "part.tar.gz"
        for part in marker.get("parts", [marker]):
            aws.get(bucket, part["key"], archive, part["sha256"])
            extract_checkpoint(archive, destination, run_id)
    if "files" in marker:
        root = destination / run_id
        for name, checksum in marker["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe snapshot manifest path")
            if digest(root / path) != checksum:
                raise ValueError("snapshot file checksum mismatch: " + name)
        for path in root.rglob("*"):
            if path.is_file() and str(path.relative_to(root)) not in marker["files"]:
                path.unlink()  # A later snapshot deleted this earlier file.
