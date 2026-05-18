from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking import load_experiment_config
from benchmarking.experiment_runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments from one top-level JSON configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment JSON file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(Path(args.config))
    paths = run_experiment(config)
    print(paths.root)


if __name__ == "__main__":
    main()
