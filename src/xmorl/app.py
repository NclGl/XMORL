from __future__ import annotations

import argparse
from pathlib import Path

from xmorl.config import load_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and inspect an interactive XMORL agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML experiment configuration (default: configs/default.yaml)",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    from xmorl.experiments import run_experiment

    run_experiment(load_config(args.config))

if __name__ == "__main__":
    main()