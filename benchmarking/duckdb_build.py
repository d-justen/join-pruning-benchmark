from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarking.experiment_config import BinarySystemConfig, GitSystemConfig, SystemConfig

DEFAULT_SHARED_DUCKDB_CHECKOUT_DIR = (Path(__file__).resolve().parent.parent / "duckdb").resolve()


@dataclass(frozen=True)
class ResolvedSystem:
    kind: str
    name: str
    binary_path: Path
    version: str
    git_commit: str | None
    repo: str | None
    ref: str | None
    build_dir: Path | None = None
    checkout_dir: Path | None = None


@dataclass(frozen=True)
class DuckDBCheckoutOptions:
    checkout_dir: Path = DEFAULT_SHARED_DUCKDB_CHECKOUT_DIR
    github_branch: str | None = None
    github_repo: str | None = None


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


def _directory_is_non_empty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _looks_like_duckdb_source_tree(path: Path) -> bool:
    return (path / "Makefile").exists() and (path / "src" / "include" / "duckdb").is_dir()


def _normalize_github_repo(repo: str) -> str:
    normalized = repo.strip()
    if normalized.startswith("https://") or normalized.startswith("git@"):
        return normalized
    normalized = normalized.removesuffix(".git")
    return f"https://github.com/{normalized}.git"


def _redacted_git_version(binary: Path) -> str:
    version = get_binary_version(binary)
    parts = version.split()
    if not parts:
        return "duckdb-local-build"
    return parts[0]


def ensure_duckdb_checkout(options: DuckDBCheckoutOptions) -> Path:
    checkout_dir = options.checkout_dir.resolve()
    if _directory_is_non_empty(checkout_dir):
        if options.github_branch is None:
            if not _looks_like_duckdb_source_tree(checkout_dir):
                raise RuntimeError(
                    f"DuckDB checkout directory exists but does not look like a DuckDB source tree: {checkout_dir}"
                )
            return checkout_dir
        git_dir = checkout_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(
                f"DuckDB checkout directory at {checkout_dir} has no .git metadata; cannot update "
                "from --duckdb-github-branch after anonymization"
            )
    else:
        if options.github_branch is None:
            raise RuntimeError(
                f"DuckDB checkout directory not found or empty: {checkout_dir}. "
                "Populate it manually or pass --duckdb-github-branch."
            )
        if options.github_repo is None:
            raise RuntimeError(
                "--duckdb-github-repo is required when --duckdb-github-branch is used to create a missing checkout"
            )
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "clone",
                "--no-checkout",
                _normalize_github_repo(options.github_repo),
                str(checkout_dir),
            ]
        )

    remote_name = "origin"
    if options.github_repo is not None:
        expected_origin = _normalize_github_repo(options.github_repo)
        current_origin = run_command(["git", "remote", "get-url", remote_name], cwd=checkout_dir)
        if current_origin != expected_origin:
            raise RuntimeError(
                f"existing checkout at {checkout_dir} points to {current_origin}, expected {expected_origin}"
            )
    run_command(["git", "fetch", remote_name, options.github_branch, "--depth", "1"], cwd=checkout_dir)
    run_command(["git", "switch", "--detach", "FETCH_HEAD"], cwd=checkout_dir)
    if not _looks_like_duckdb_source_tree(checkout_dir):
        raise RuntimeError(f"DuckDB checkout directory is incomplete after update: {checkout_dir}")
    return checkout_dir


def ensure_duckdb_vcpkg_toolchain(clone_dir: Path, env: dict[str, str]) -> Path:
    toolchain_path = clone_dir / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake"
    if toolchain_path.exists():
        return toolchain_path.resolve()
    run_command(["make", "setup-vcpkg"], cwd=clone_dir, env=env)
    if not toolchain_path.exists():
        raise FileNotFoundError(f"DuckDB vcpkg toolchain not found after setup: {toolchain_path}")
    return toolchain_path.resolve()


def build_duckdb(
    clone_dir: Path,
    config: GitSystemConfig,
    required_extensions: frozenset[str] = frozenset(),
) -> Path:
    build_env = dict(os.environ)
    build_env["GEN"] = "ninja"
    build_env["CCACHE_DISABLE"] = "1"
    if config.c_compiler is not None:
        build_env["CC"] = config.c_compiler
    if config.cxx_compiler is not None:
        build_env["CXX"] = config.cxx_compiler
    if "ducklake" not in required_extensions:
        # Keep native paper builds separate from build/release. DuckLake builds
        # configure that directory with a vcpkg toolchain, which CMake retains
        # on later plain `make` invocations and can unexpectedly rebuild curl.
        build_dir = clone_dir / "build" / "paper-native-release"
        build_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "cmake",
                "-G",
                "Ninja",
                "-DFORCE_COLORED_OUTPUT=1",
                '-DLOCAL_EXTENSION_REPO=',
                '-DOVERRIDE_GIT_DESCRIBE=',
                '-DDUCKDB_EXPLICIT_VERSION=',
                # Native paper workloads use DuckDB database files. Skipping
                # Parquet also keeps the older RPT+ fork compatible with newer
                # libc++, whose iterator requirements reject its bundled
                # Thrift-generated enum iterator.
                "-DSKIP_EXTENSIONS=parquet",
                "-DCMAKE_BUILD_TYPE=Release",
                "../..",
            ],
            cwd=build_dir,
            env=build_env,
        )
        run_command(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_dir,
            env=build_env,
        )
        return build_dir

    extension_configs = [
        clone_dir / ".github" / "config" / "extensions" / "ducklake.cmake"
    ]
    if "httpfs" in required_extensions:
        extension_configs.append(
            clone_dir / ".github" / "config" / "extensions" / "httpfs.cmake"
        )
    for extension_config in extension_configs:
        if not extension_config.exists():
            raise FileNotFoundError(f"extension config not found: {extension_config}")
    toolchain_path = ensure_duckdb_vcpkg_toolchain(clone_dir, build_env)
    build_env["DUCKDB_NEW_EXTENSION_BUILD"] = "1"
    build_env["EXTENSION_CONFIGS"] = ";".join(
        str(extension_config) for extension_config in extension_configs
    )
    build_env["VCPKG_TOOLCHAIN_PATH"] = str(toolchain_path)
    run_command(["make", "sync_out_of_tree_extensions"], cwd=clone_dir, env=build_env)
    run_command(["make", "release"], cwd=clone_dir, env=build_env)
    return clone_dir / "build" / "release"


def resolve_built_duckdb_binary(
    clone_dir: Path,
    preferred_build_dir: Path | None = None,
) -> Path:
    candidates = []
    if preferred_build_dir is not None:
        candidates.append(preferred_build_dir / "duckdb")
    candidates.extend([
        clone_dir / "build" / "release" / "duckdb",
        clone_dir / "build" / "debug" / "duckdb",
        clone_dir / "build" / "duckdb",
        clone_dir / "duckdb",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate compiled DuckDB CLI after build; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def get_git_commit(clone_dir: Path) -> str:
    return run_command(["git", "rev-parse", "HEAD"], cwd=clone_dir)


def _validate_required_extensions(
    binary_path: Path,
    required_extensions: frozenset[str],
) -> None:
    for extension_name in sorted(required_extensions):
        run_command(
            [str(binary_path), ":memory:", "-c", f"LOAD {extension_name};"],
        )


def _resolve_binary_system(
    config: BinarySystemConfig,
    required_extensions: frozenset[str] = frozenset(),
) -> ResolvedSystem:
    binary_path = config.binary_path
    if binary_path.is_absolute() or binary_path.parent != Path("."):
        binary_path = binary_path.resolve()
    if not binary_path.exists():
        candidate = shutil.which(str(config.binary_path))
        if candidate is None:
            raise FileNotFoundError(f"DuckDB binary not found: {config.binary_path}")
        binary_path = Path(candidate).resolve()
    _validate_required_extensions(binary_path, required_extensions)
    return ResolvedSystem(
        kind="binary",
        name=config.name,
        binary_path=binary_path,
        version=get_binary_version(binary_path),
        git_commit=None,
        repo=None,
        ref=None,
    )


def _resolve_git_system(
    config: GitSystemConfig,
    checkout_options: DuckDBCheckoutOptions,
    required_extensions: frozenset[str] = frozenset(),
) -> ResolvedSystem:
    uses_shared_checkout = config.checkout_dir is None
    effective_options = (
        checkout_options
        if uses_shared_checkout
        else DuckDBCheckoutOptions(checkout_dir=config.checkout_dir)
    )
    checkout_dir = ensure_duckdb_checkout(effective_options)
    if uses_shared_checkout:
        apply_compatibility_patches(checkout_dir)
    build_dir = build_duckdb(checkout_dir, config, required_extensions)
    binary_path = resolve_built_duckdb_binary(checkout_dir, build_dir)
    _validate_required_extensions(binary_path, required_extensions)
    return ResolvedSystem(
        kind="git",
        name=config.name,
        binary_path=binary_path,
        version=_redacted_git_version(binary_path),
        git_commit=None,
        repo=None,
        ref=None,
        build_dir=build_dir.resolve(),
        checkout_dir=checkout_dir.resolve(),
    )


def resolve_system(
    config: SystemConfig,
    system_dir: Path,
    checkout_options: DuckDBCheckoutOptions | None = None,
    required_extensions: frozenset[str] = frozenset(),
) -> ResolvedSystem:
    if config.kind == "binary":
        return _resolve_binary_system(config, required_extensions)
    return _resolve_git_system(
        config,
        checkout_options or DuckDBCheckoutOptions(),
        required_extensions,
    )


def resolve_existing_system(
    config: SystemConfig,
    system_dir: Path,
    checkout_options: DuckDBCheckoutOptions | None = None,
) -> ResolvedSystem:
    if config.kind == "binary":
        return _resolve_binary_system(config)
    checkout_dir = (
        config.checkout_dir
        if config.checkout_dir is not None
        else (checkout_options or DuckDBCheckoutOptions()).checkout_dir
    ).resolve()
    if not _looks_like_duckdb_source_tree(checkout_dir):
        raise RuntimeError(f"DuckDB checkout directory does not look usable: {checkout_dir}")
    binary_path = resolve_built_duckdb_binary(checkout_dir)
    return ResolvedSystem(
        kind="git",
        name=config.name,
        binary_path=binary_path,
        version=_redacted_git_version(binary_path),
        git_commit=None,
        repo=None,
        ref=None,
    )
