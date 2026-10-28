"""Upload and remove only this EC2 attempt's disposable benchmark objects."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path, PurePosixPath


class ScratchData:
    def __init__(self, aws, bucket: str, run: str, attempt: str, *, data_root=Path('/mnt/benchmark'), deadline=None):
        for name in (run, attempt):
            self._name(name)
        self.aws, self.bucket = aws, bucket
        self.prefix = f"scratch/{run}/{attempt}/"
        self.data_root = Path(data_root).resolve()
        self.deadline = deadline

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("unsafe scratch namespace")

    def objects(self, prefix):
        token = None
        while True:
            args = ["--bucket", self.bucket, "--prefix", prefix, "--no-paginate"]
            if token:
                args += ["--continuation-token", token]
            result = self.aws.command("s3api", "list-objects-v2", *args)
            yield from result.get("Contents", [])
            if not result.get("IsTruncated"):
                break
            token = result["NextContinuationToken"]

    def delete(self, workload=None):
        if workload is not None:
            self._name(workload)
        prefix = self.prefix + (workload + "/" if workload is not None else "")
        # Snapshot the list before deletion: continuation tokens must not cross mutations.
        keys = [item["Key"] for item in self.objects(prefix)]
        for offset in range(0, len(keys), 1000):
            result = self.aws.command("s3api", "delete-objects", payload={"Bucket": self.bucket,
                "Delete": {"Objects": [{"Key": key} for key in keys[offset:offset+1000]], "Quiet": True}})
            if result.get("Errors"):
                raise RuntimeError(f"S3 scratch deletion failed: {result['Errors']}")
        if list(self.objects(prefix)):
            raise RuntimeError("S3 scratch cleanup left objects behind")
        return {"deleted_objects": len(keys), "prefix": prefix, "verified_empty": True}

    def upload(self, workload, catalog):
        self._name(workload)
        relative = PurePosixPath(catalog)
        if not relative.is_relative_to('/work') or '..' in relative.parts:
            raise ValueError("S3 upload catalog must be inside /work")
        local = (self.data_root / relative.relative_to('/work')).resolve()
        if not local.is_relative_to(self.data_root):
            raise ValueError("S3 upload catalog escapes the data volume")
        versioning = self.aws.command("s3api", "get-bucket-versioning", "--bucket", self.bucket)
        if versioning.get("Status") in ("Enabled", "Suspended"):
            raise ValueError("DuckLake scratch bucket must never have versioning enabled")
        files = json.loads(local.read_text())
        root = (local.parent / 'parquet').resolve()
        if not root.is_relative_to(self.data_root):
            raise ValueError("Parquet directory escapes the data volume")
        prefix = self.prefix + workload + '/'
        self.delete(workload)
        uploaded = []
        try:
            for table, paths in sorted(files.items()):
                for name in paths:
                    path = (root / name).resolve()
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
                        raise ValueError("unsafe Parquet catalog path")
                    if self.deadline and time.time() >= self.deadline:
                        raise TimeoutError("deadline reached during S3 upload")
                    result = self.aws.put(self.bucket, prefix + name, path)
                    uploaded.append({"table": table, "key": prefix + name, "bytes": path.stat().st_size,
                                     "sha256_base64": result["ChecksumSHA256"]})
        except BaseException:
            self.delete(workload)
            raise
        return {"base_url": f"s3://{self.bucket}/{prefix.rstrip('/')}", "objects": uploaded}
