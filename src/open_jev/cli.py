from __future__ import annotations

import argparse
import json

from .runner import run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="open-jev")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "train-synthetic", help="train the baseline on deterministic synthetic data"
    )
    train.add_argument("--output", default="artifacts/synthetic-run")
    train.add_argument("--groups", type=int, default=600)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.command == "train-synthetic":
        metrics = run_synthetic_training(
            args.output,
            groups=args.groups,
            epochs=args.epochs,
            seed=args.seed,
            hidden_size=args.hidden_size,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(metrics, indent=2))
    return 0
