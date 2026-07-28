from __future__ import annotations

import argparse
from pathlib import Path

from xmorl.config import load_config

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for selecting an experiment configuration

    :return: Configured command-line parser
    """
    parser = argparse.ArgumentParser(description="Train and inspect an interactive XMORL agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sdst.yaml"),
        help="YAML experiment configuration (default: configs/sdst.yaml)",
    )
    return parser

def main() -> None:
    """
    Load the requested configuration and run its experiment

    :return: None
    """
    args = build_parser().parse_args()
    from xmorl.experiments import run_experiment

    run_experiment(load_config(args.config))

if __name__ == "__main__":
    main()
