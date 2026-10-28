"""Resolve a paper suite to a portable, self-contained launch configuration.

This module uses only the standard library: the host launcher needs no Python
benchmark dependencies. Referenced configs are embedded before leaving the host.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path


def load_plan(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if "groups" not in raw:
        return raw
    raw = copy.deepcopy(raw)
    for group in raw["groups"]:
        config = group["config"]
        if isinstance(config, str):
            config = json.loads((path.parent / config).read_text())
        if "groups" in config:
            raise ValueError("nested paper suites are not supported")
        overrides = group.pop("binary_paths", {})
        systems = {s["name"]: s for s in config["systems"]}
        if overrides.keys() - systems.keys():
            raise ValueError("binary override names an unknown system")
        for name, binary in overrides.items():
            systems[name]["binary_path"] = binary
        group["config"] = config
    validate_plan(raw)
    return raw


def memory_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt])(?:i?b)?", str(value).lower())
    if not match:
        raise ValueError("unsupported container memory limit: " + str(value))
    return int(float(match[1]) * 1024 ** ("kmgt".index(match[2]) + 1))


def validate_plan(raw: dict) -> None:
    groups = raw["groups"]
    names = [g["name"] for g in groups]
    if not names or len(names) != len(set(names)):
        raise ValueError("paper suite groups must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", n) for n in names):
        raise ValueError("unsafe paper suite group name")
    limits = raw["suite"]
    if limits["container_cpus"] <= 0:
        raise ValueError("container CPUs must be positive")
    for config in experiment_configs(raw):
        if "groups" in config:
            raise ValueError("nested paper suites are not supported")
        if config["suite"]["container_cpus"] > limits["container_cpus"]:
            raise ValueError("group CPUs exceed paper suite container limit")
        if memory_bytes(config["suite"]["container_memory"]) > memory_bytes(limits["container_memory"]):
            raise ValueError("group memory exceeds paper suite container limit")


def experiment_configs(raw: dict) -> list[dict]:
    return [g["config"] for g in raw["groups"]] if "groups" in raw else [raw]


def uses_s3(raw: dict) -> bool:
    return any(w.get("params", {}).get("storage", {}).get("data_source") == "s3"
               for config in experiment_configs(raw) for w in config["workloads"])
