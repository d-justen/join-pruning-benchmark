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


def apply_compatibility_patches(clone_dir: Path) -> None:
    enum_path = clone_dir / "src" / "include" / "duckdb" / "execution" / "operator" / "join" / "join_filter_pushdown.hpp"
    join_path = clone_dir / "src" / "execution" / "operator" / "join" / "physical_hash_join.cpp"
    if not enum_path.exists() or not join_path.exists():
        return

    enum_text = enum_path.read_text(encoding="utf-8")
    join_text = join_path.read_text(encoding="utf-8")

    needs_enum_patch = "MIN_MAX,\n\tBLOOM,\n\tMIN_MAX_AND_BLOOM," not in enum_text
    buggy_plan_branch = (
        "result.type = settings.enable_min_max_filter_pushdown ? JoinFilterSummaryPlanType::MIN_MAX_AND_BLOOM\n"
        "\t\t\t                                                      : JoinFilterSummaryPlanType::MIN_MAX_AND_BLOOM;"
    )
    fixed_plan_branch = (
        "result.type = settings.enable_min_max_filter_pushdown ? JoinFilterSummaryPlanType::MIN_MAX_AND_BLOOM\n"
        "\t\t\t                                                      : JoinFilterSummaryPlanType::BLOOM;"
    )
    needs_join_patch = buggy_plan_branch in join_text and "case JoinFilterSummaryPlanType::BLOOM:" not in join_text

    if not needs_enum_patch and not needs_join_patch:
        return

    if needs_enum_patch:
        enum_needle = "enum class JoinFilterSummaryPlanType : uint8_t {\n\tNONE,\n\tSINGLE_VALUE,\n\tMIN_MAX,\n\tMIN_MAX_AND_BLOOM,"
        enum_replacement = "enum class JoinFilterSummaryPlanType : uint8_t {\n\tNONE,\n\tSINGLE_VALUE,\n\tMIN_MAX,\n\tBLOOM,\n\tMIN_MAX_AND_BLOOM,"
        if enum_needle not in enum_text:
            raise RuntimeError(f"failed to apply DuckDB compatibility patch: enum shape not found in {enum_path}")
        enum_text = enum_text.replace(enum_needle, enum_replacement, 1)
        enum_path.write_text(enum_text, encoding="utf-8")

    if needs_join_patch:
        if buggy_plan_branch not in join_text:
            raise RuntimeError(f"failed to apply DuckDB compatibility patch: buggy branch not found in {join_path}")
        join_text = join_text.replace(buggy_plan_branch, fixed_plan_branch, 1)
        switch_needle = (
            "\t\t\tcase JoinFilterSummaryPlanType::MIN_MAX:\n"
            "\t\t\t\tCreateDynamicMinMaxFilters(op, info, filter_col_idx, cmp, min_val, max_val);\n"
            "\t\t\t\tbreak;\n"
            "\t\t\tcase JoinFilterSummaryPlanType::MIN_MAX_AND_BLOOM:\n"
        )
        switch_replacement = (
            "\t\t\tcase JoinFilterSummaryPlanType::MIN_MAX:\n"
            "\t\t\t\tCreateDynamicMinMaxFilters(op, info, filter_col_idx, cmp, min_val, max_val);\n"
            "\t\t\t\tbreak;\n"
            "\t\t\tcase JoinFilterSummaryPlanType::BLOOM:\n"
            "\t\t\t\tD_ASSERT(runtime_filter_ht);\n"
            "\t\t\t\tPushBloomFilter(op, *runtime_filter_ht, info, filter_col_idx);\n"
            "\t\t\t\tbreak;\n"
            "\t\t\tcase JoinFilterSummaryPlanType::MIN_MAX_AND_BLOOM:\n"
        )
        if switch_needle not in join_text:
            raise RuntimeError(f"failed to apply DuckDB compatibility patch: switch shape not found in {join_path}")
        join_text = join_text.replace(switch_needle, switch_replacement, 1)
        join_path.write_text(join_text, encoding="utf-8")


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
    apply_compatibility_patches(checkout_dir)
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
