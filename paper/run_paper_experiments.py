from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking import load_experiment_config
from benchmarking.experiment_runner import run_experiment
from paper.figures.build_all import build_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local paper experiment suite and generate paper-ready PDF figures."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper/configs/local_paper_experiments.json"),
        help="path to the paper experiment config",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="reuse an existing experiment root and only regenerate figures",
    )
    execution.add_argument(
        "--only-microbenchmarks",
        action="store_true",
        help="run configured filter/JPB microbenchmarks and regenerate figures, preserving existing SQL results",
    )
    execution.add_argument(
        "--resume-preparation",
        action="store_true",
        help="continue failed preparation, reusing completed workloads and density dimensions/batches",
    )
    parser.add_argument(
        "--system-name",
        help="visualize one system instead of combining all configured systems",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    experiment_root = config.experiment_root
    if not args.skip_benchmarks:
        experiment_root = run_experiment(config, resume_preparation=args.resume_preparation,
                                         only_microbenchmarks=args.only_microbenchmarks).root
    output_dir = build_figures(experiment_root, args.system_name)
    print(output_dir)


if __name__ == "__main__":
    main()
