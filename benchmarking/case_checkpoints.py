"""Verified subprocess checkpoints for individual microbenchmark cases."""
import csv
import json
import subprocess
from pathlib import Path

from benchmarking.suite_io import fingerprint, sha256, write_json


def run_case(command, output: Path, identity, callback=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_suffix(".complete.json")
    identity = fingerprint(identity)
    if marker.exists() and output.exists():
        saved = json.loads(marker.read_text())
        if saved.get("identity") == identity and saved.get("sha256") == sha256(output):
            return
    temporary = output.with_suffix(".tmp")
    with output.with_suffix(".log").open("w") as log:
        subprocess.run([*command, "--output=" + str(temporary.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    # CSV semantic validation is done against the full expected case set after
    # assembly; incomplete commands never receive a checkpoint marker.
    with temporary.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("microbenchmark case produced no rows")
    temporary.replace(output)
    write_json(marker, {"identity": identity, "sha256": sha256(output)})
    if callback:
        callback()


def combine(paths, destination):
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="") as out:
        writer = None
        for path in paths:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                if reader.fieldnames != writer.fieldnames:
                    raise ValueError("microbenchmark case CSV schemas differ")
                writer.writerows(reader)
    temporary.replace(destination)
