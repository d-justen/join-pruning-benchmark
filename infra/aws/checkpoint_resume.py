"""Compatibility for attempt-specific S3 identities in already-pinned images."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from infra.aws.common import digest, write_json
except ModuleNotFoundError:  # Standalone EC2 worker bundle.
    from common import digest, write_json


def legacy_identity(payload):
    value = {k: v for k, v in payload.items() if k != "existing_only"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rebind_legacy_ducklake_units(run: Path, bucket: str, attempt: str) -> int:
    """Rebind verified checkpoints on the restored copy, preserving their history.

    Resume deliberately retains the original image, runner and query binaries.
    Its old runner needs the *new* staging URL in the checkpoint identity to skip
    completed units. New runners use URL-independent identities and need no repair.
    This only changes scheduling metadata, never results, SQL, profiles or traces.
    """
    run = run.resolve()
    safe = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
    if not all(safe.fullmatch(value) for value in (run.name, attempt)):
        raise ValueError("unsafe resume namespace")
    count = 0
    for status_path in sorted((run / "units").glob("*/status.json")):
        state = json.loads(status_path.read_text())
        if state.get("status") != "complete" or state.get("action") != "measure":
            continue
        payload_path = status_path.with_name("payload.json")
        if not payload_path.is_file():
            continue
        payload = json.loads(payload_path.read_text())
        workload = payload.get("workload")
        if not isinstance(workload, dict):
            continue
        storage = workload.get("params", {}).get("storage", {})
        if storage.get("kind") != "ducklake" or storage.get("data_source") != "s3":
            continue
        name = workload.get("name", "")
        if not safe.fullmatch(name) or payload.get("action") != "measure" or state.get("workload") != name:
            continue
        if state.get("identity") != legacy_identity(payload):
            continue  # Already normalized, or the saved payload is not authentic.
        old_url = payload.get("s3_base_url", "")
        prefix = f"s3://{bucket}/scratch/{run.name}/"
        if not isinstance(old_url, str) or not old_url.startswith(prefix):
            continue
        parts = old_url[len(prefix):].split("/")
        if len(parts) != 2 or not safe.fullmatch(parts[0]) or parts[1] != name:
            continue
        new_url = f"{prefix}{attempt}/{name}"
        if old_url == new_url:
            continue
        database = status_path.with_name("results.duckdb")
        if not database.is_file() or digest(database) != state.get("sha256"):
            continue
        valid = True
        for relative, checksum in state.get("artifacts", {}).items():
            path = (run / relative).resolve()
            if not path.is_relative_to(run) or not path.is_file() or digest(path) != checksum:
                valid = False
                break
        if not valid:
            continue
        rebound = {**payload, "s3_base_url": new_url}
        identity = legacy_identity(rebound)
        evidence = run / "cloud" / attempt / "ducklake-resume" / status_path.parent.name
        # Keep the original metadata before rebinding the restored scheduling
        # records. The original S3 checkpoint itself remains immutable.
        write_json(evidence / "payload.json", payload)
        write_json(evidence / "status.json", state)
        write_json(evidence / "rebind.json", {"old_url": old_url, "new_url": new_url,
                   "old_identity": state["identity"], "new_identity": identity})
        write_json(payload_path, rebound)
        write_json(status_path, {**state, "identity": identity})
        count += 1
    return count
