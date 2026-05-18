from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.results_analysis import analyze_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results, backfill profiling if needed, and create comparison charts."
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help="path to an experiment directory under out/experiments/",
    )
    parser.add_argument(
        "--baseline-variant",
        help="override the baseline variant label used for comparisons",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = analyze_experiment(args.experiment_root, args.baseline_variant)
    print(paths.root)


if __name__ == "__main__":
    main()
