"""An auditable synthetic training experiment with held-out evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from .batching import candidate_labels
from .data import dataset_fingerprint, split_examples, write_jsonl
from .metrics import accuracy, brier_score, expected_calibration_error
from .model import DynamicDecisionModel
from .provenance import build_receipt
from .schema import Example
from .synthetic import generate_synthetic_dataset
from .tokenization import WhitespaceTokenizer
from .training import Trainer, TrainingConfig


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _evaluate(predictions: list[dict], rows: list[Example]) -> dict:
    width = max(len(row.target()) for row in rows)
    probabilities = torch.zeros(len(rows), width)
    targets = torch.zeros_like(probabilities)
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    for i, (prediction, row) in enumerate(zip(predictions, rows, strict=True)):
        labels = candidate_labels(row)
        probabilities[i, : len(labels)] = torch.tensor(
            [prediction["probabilities"][label] for label in labels]
        )
        targets[i, : len(labels)] = torch.tensor(row.target())
        mask[i, : len(labels)] = True
    return {
        "count": len(rows),
        "accuracy": accuracy(probabilities, targets),
        "nll": float(-(targets * probabilities.clamp_min(1e-12).log()).sum(-1).mean()),
        "brier": brier_score(probabilities, targets, mask),
        "ece": expected_calibration_error(probabilities, targets),
    }


def _majority_baseline(train: list[Example], test: list[Example]) -> dict:
    by_type = {}
    correct = 0
    for kind in ("choice", "noul", "score"):
        counts = Counter(row.label for row in train if row.question.type == kind)
        majority = counts.most_common(1)[0][0]
        selected = [row for row in test if row.question.type == kind]
        hits = sum(row.label == majority for row in selected)
        correct += hits
        by_type[kind] = {
            "label": majority,
            "count": len(selected),
            "accuracy": hits / len(selected),
        }
    return {"accuracy": correct / len(test), "by_type": by_type}


def run_synthetic_training(
    output_dir: str | Path,
    *,
    groups: int = 600,
    epochs: int = 20,
    seed: int = 7,
    hidden_size: int = 64,
    device: str = "auto",
    batch_size: int = 32,
    learning_rate: float = 3e-3,
) -> dict[str, object]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    rows = generate_synthetic_dataset(groups=groups, seed=seed)
    splits = split_examples(rows, seed=seed)
    for name, values in splits.items():
        write_jsonl(values, output / f"{name}.jsonl")
    split_files = [output / f"{name}.jsonl" for name in splits]
    config = TrainingConfig(
        epochs=epochs, seed=seed, device=device, batch_size=batch_size, learning_rate=learning_rate
    )
    torch.manual_seed(seed)  # Includes initialization, not just the subsequent batch order.
    tokenizer = WhitespaceTokenizer.fit(splits["train"])
    trainer = Trainer(
        DynamicDecisionModel(len(tokenizer), hidden_size=hidden_size), tokenizer, config
    )
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_sha256 = hashlib.sha256(
        b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in source_files)
    ).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    manifest = {
        "dataset": "synthetic-v2",
        "seed": seed,
        "groups": groups,
        "examples": len(rows),
        "fingerprint": dataset_fingerprint(rows),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_fingerprints": {
            name: dataset_fingerprint(values) for name, values in splits.items()
        },
        "calibration_used": False,
        "config": {**asdict(config), "hidden_size": hidden_size},
        "started_at_utc": started,
        "source_sha256": source_sha256,
        "file_receipt": build_receipt(
            root=output,
            files=split_files,
            sources=[
                {
                    "name": "open-jev synthetic-v2",
                    "url": "https://github.com/Eggwardhan/open-jev",
                    "role": "auditable training fixture",
                }
            ],
        ),
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    }
    _write_json(output / "manifest.json", manifest)
    untrained_predictions = trainer.predict(splits["test"])
    untrained = _evaluate(untrained_predictions, splits["test"])
    history_path = output / "history.jsonl"

    def log_epoch(row: dict[str, float]) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    start = time.perf_counter()
    history = trainer.fit(splits["train"], splits["validation"], on_epoch=log_epoch)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    training_seconds = time.perf_counter() - start
    trainer.save(output / "model.pt")
    predictions = trainer.predict(splits["test"])
    restored = Trainer.load(output / "model.pt", device=str(trainer.device))
    reloaded_predictions = restored.predict(splits["test"])
    reload_error = max(
        abs(p["probabilities"][key] - q["probabilities"][key])
        for p, q in zip(predictions, reloaded_predictions, strict=True)
        for key in p["probabilities"]
    )
    if reload_error > 1e-6:
        raise RuntimeError(f"checkpoint reload mismatch: {reload_error}")
    final = _evaluate(predictions, splits["test"])
    by_type = {}
    for kind in ("choice", "noul", "score"):
        pairs = [
            (prediction, row)
            for prediction, row in zip(predictions, splits["test"], strict=True)
            if row.question.type == kind
        ]
        by_type[kind] = _evaluate([p for p, _ in pairs], [r for _, r in pairs])
        if kind == "score":
            by_type[kind]["score_mae"] = sum(
                abs(
                    sum(
                        index * p["probabilities"][label]
                        for index, label in enumerate(candidate_labels(row))
                    )
                    - row.label
                )
                for p, row in pairs
            ) / len(pairs)
    metrics = {
        "dataset_fingerprint": dataset_fingerprint(rows),
        **{f"{name}_examples": len(values) for name, values in splits.items()},
        **{f"test_{key}": value for key, value in final.items() if key != "count"},
        "untrained": untrained,
        "majority_baseline": _majority_baseline(splits["train"], splits["test"]),
        "by_type": by_type,
        "device": str(trainer.device),
        "gpu_name": torch.cuda.get_device_name(trainer.device)
        if trainer.device.type == "cuda"
        else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device)
        if trainer.device.type == "cuda"
        else 0,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "training_seconds": training_seconds,
        "selected_epoch": trainer.selected_epoch,
        "reload_max_abs_error": reload_error,
    }
    _write_json(output / "history.json", history)
    _write_json(
        output / "predictions.json",
        [{**p, "target": row.label} for p, row in zip(predictions, splits["test"], strict=True)],
    )
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    )
    return metrics
