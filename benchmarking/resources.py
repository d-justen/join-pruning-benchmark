"""Resource policy shared by every preparation connection in suite workers."""
from __future__ import annotations

import json
import os
import re

import duckdb


def size_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(i?B)?\s*", str(value), re.I)
    if not match:
        raise ValueError(f"invalid resource size: {value}")
    power = " KMGT".index(match[2].upper()) if match[2] else 0
    base = 1024 if (match[3] or "").lower().startswith("i") or not match[3] else 1000
    return int(float(match[1]) * base ** power)


def connect(*args, **kwargs):
    options = dict(kwargs.pop("config", {}) or {})
    # Absent outside suite workers: the legacy entry points keep their defaults.
    policy = json.loads(os.environ.get("BENCH_PREPARATION_CONFIG", "{}"))
    options.update(policy)
    return duckdb.connect(*args, config=options, **kwargs)
