"""Use the invoking user's identity for durable bind-mounted artifacts."""
import os
import sys
from pathlib import Path

uid = int(os.environ.get("BENCH_UID", os.getuid()))
gid = int(os.environ.get("BENCH_GID", os.getgid()))
if os.geteuid() == 0 and uid != 0:
    for directory in (Path("/work"), Path("/results")):
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, uid, gid)
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
os.execv(sys.executable, [sys.executable, "-m", "benchmarking.suite", *sys.argv[1:]])
