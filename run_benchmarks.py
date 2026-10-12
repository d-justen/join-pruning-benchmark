from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking import load_experiment_config
from benchmarking.duckdb_build import DEFAULT_SHARED_DUCKDB_CHECKOUT_DIR, DuckDBCheckoutOptions
from benchmarking.experiment_runner import run_experiment_with_options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments from one top-level JSON configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment JSON file",
    )
    parser.add_argument(
        "--duckdb-checkout-dir",
        type=Path,
        default=DEFAULT_SHARED_DUCKDB_CHECKOUT_DIR,
        help=f"path to the shared local DuckDB source tree (default: {DEFAULT_SHARED_DUCKDB_CHECKOUT_DIR})",
    )
    parser.add_argument(
        "--duckdb-github-branch",
        help="runtime-only GitHub branch to fetch into the shared DuckDB checkout",
    )
    parser.add_argument(
        "--duckdb-github-repo",
        help="runtime-only GitHub repo used only when creating or updating the shared DuckDB checkout",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--resume-preparation",
        action="store_true",
        help="continue failed preparation, reusing completed workloads and density dimensions/batches",
    )
    execution.add_argument(
        "--only-microbenchmarks", action="store_true",
        help="run configured filter/JPB microbenchmarks while preserving existing SQL results",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(Path(args.config))
    paths = run_experiment_with_options(
        config,
        DuckDBCheckoutOptions(
            checkout_dir=args.duckdb_checkout_dir.resolve(),
            github_branch=args.duckdb_github_branch,
            github_repo=args.duckdb_github_repo,
        ),
        resume_preparation=args.resume_preparation,
        only_microbenchmarks=args.only_microbenchmarks,
    )
    print(paths.root)


if __name__ == "__main__":
    main()
