"""Small durable I/O primitives used by the container workflow."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def tree_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except FileNotFoundError:
            pass  # Spill files may disappear between listing and stat.
    return total


def disk_preflight(path: Path, required_gib: float, reserve_gib: float) -> None:
    free = shutil.disk_usage(path).free
    needed = int((required_gib + reserve_gib) * 1024**3)
    if free < needed:
        raise RuntimeError(
            f"insufficient space at {path}: {free / 1024**3:.1f} GiB free, "
            f"{required_gib:.1f} GiB estimated peak + {reserve_gib:.1f} GiB reserve required"
        )
