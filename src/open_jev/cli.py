from __future__ import annotations

import argparse
import json

from .runner import run_synthetic_training
from .local_model import DEFAULT_LOCAL_MODEL, LocalDecisionModel
from .backends import load_backend
from .schema import Question


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
    local = subparsers.add_parser(
        "local-decide", help="run one decision with a local HF causal model"
    )
    local.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    local.add_argument("--state", required=True)
    local.add_argument("--type", choices=("choice", "noul", "score"), required=True)
    local.add_argument("--instructions", required=True)
    local.add_argument(
        "--criteria", default=None, help="JSON object for choice or JSON array for score"
    )
    local.add_argument("--device", default="auto")
    local.add_argument("--backend", choices=("causal", "encoder", "sglang"), default="causal")
    local.add_argument(
        "--endpoint", default=None, help="local SGLang endpoint when --backend sglang"
    )
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
    elif args.command == "local-decide":
        criteria = json.loads(args.criteria) if args.criteria is not None else None
        question = Question(type=args.type, instructions=args.instructions, criteria=criteria)
        if args.backend == "causal":
            model = LocalDecisionModel.from_pretrained(args.model, device=args.device)
        else:
            model = load_backend(args.model, endpoint=args.endpoint, device=args.device)
        print(json.dumps(model.decide(args.state, question), indent=2, ensure_ascii=False))
    return 0
