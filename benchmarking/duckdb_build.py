from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarking.experiment_config import BinarySystemConfig, GitSystemConfig, SystemConfig


@dataclass(frozen=True)
class ResolvedSystem:
    kind: str
    name: str
    binary_path: Path
    version: str
    git_commit: str | None
    repo: str | None
    ref: str | None


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cwd: {cwd or Path.cwd()}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def get_binary_version(binary: Path) -> str:
    return run_command([str(binary), "--version"])


def ensure_duckdb_checkout(clone_dir: Path, repo: str, ref: str) -> None:
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", "--no-checkout", repo, str(clone_dir)])
    git_dir = clone_dir / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"existing clone path is not a git checkout: {clone_dir}")
    current_origin = run_command(["git", "remote", "get-url", "origin"], cwd=clone_dir)
    if current_origin != repo:
        raise RuntimeError(
            f"existing clone at {clone_dir} points to {current_origin}, expected {repo}"
        )
    run_command(["git", "fetch", "origin", ref, "--depth", "1"], cwd=clone_dir)
    run_command(["git", "switch", "--detach", "FETCH_HEAD"], cwd=clone_dir)


def build_duckdb(clone_dir: Path) -> None:
    build_env = dict(os.environ)
    build_env["GEN"] = "ninja"
    run_command(["make"], cwd=clone_dir, env=build_env)


def resolve_built_duckdb_binary(clone_dir: Path) -> Path:
    candidates = [
        clone_dir / "build" / "release" / "duckdb",
        clone_dir / "build" / "debug" / "duckdb",
        clone_dir / "build" / "duckdb",
        clone_dir / "duckdb",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate compiled DuckDB CLI after build; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def get_git_commit(clone_dir: Path) -> str:
    return run_command(["git", "rev-parse", "HEAD"], cwd=clone_dir)


def _resolve_binary_system(config: BinarySystemConfig) -> ResolvedSystem:
    binary_path = config.binary_path
    if binary_path.is_absolute() or binary_path.parent != Path("."):
        binary_path = binary_path.resolve()
    if not binary_path.exists():
        candidate = shutil.which(str(config.binary_path))
        if candidate is None:
            raise FileNotFoundError(f"DuckDB binary not found: {config.binary_path}")
        binary_path = Path(candidate).resolve()
    return ResolvedSystem(
        kind="binary",
        name=config.name,
        binary_path=binary_path,
        version=get_binary_version(binary_path),
        git_commit=None,
        repo=None,
        ref=None,
    )


def _resolve_git_system(config: GitSystemConfig, system_dir: Path) -> ResolvedSystem:
    checkout_dir = system_dir / "checkout"
    ensure_duckdb_checkout(checkout_dir, config.repo, config.ref)
    build_duckdb(checkout_dir)
    binary_path = resolve_built_duckdb_binary(checkout_dir)
    return ResolvedSystem(
        kind="git",
        name=config.name,
        binary_path=binary_path,
        version=get_binary_version(binary_path),
        git_commit=get_git_commit(checkout_dir),
        repo=config.repo,
        ref=config.ref,
    )


def resolve_system(config: SystemConfig, system_dir: Path) -> ResolvedSystem:
    if config.kind == "binary":
        return _resolve_binary_system(config)
    return _resolve_git_system(config, system_dir)


def resolve_existing_system(config: SystemConfig, system_dir: Path) -> ResolvedSystem:
    if config.kind == "binary":
        return _resolve_binary_system(config)
    checkout_dir = system_dir / "checkout"
    binary_path = resolve_built_duckdb_binary(checkout_dir)
    return ResolvedSystem(
        kind="git",
        name=config.name,
        binary_path=binary_path,
        version=get_binary_version(binary_path),
        git_commit=get_git_commit(checkout_dir),
        repo=config.repo,
        ref=config.ref,
    )
