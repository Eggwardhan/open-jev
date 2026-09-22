# open-jev

[中文文档](README.zh-CN.md)

> **Typed, calibrated decisions for AI agents.**

open-jev is an open-source PyTorch decision layer for agent routing, tool
selection, RAG checks, and evaluation. It turns a state, a typed question, and
a dynamic candidate set into probabilities, evidence, and replayable outputs.

[![CI](https://github.com/Eggwardhan/open-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/Eggwardhan/open-jev/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## Why open-jev?

- **Typed decisions** over dynamic candidate sets
- **Calibrated probabilities** instead of unqualified labels
- **Auditable runs** with provenance, evidence, and replay keys
- **Replaceable components** for tokenizers, encoders, and scoring heads
- **Reproducible evaluation** across accuracy, calibration, latency, and cost

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Run the included deterministic training experiment:

```bash
open-jev train-synthetic --output artifacts/my-run \
  --groups 600 --epochs 40 --batch-size 32 --seed 7 --device auto
```

Use `--device cuda` to require CUDA or `--device cpu` to run locally. The
runner writes grouped JSONL splits, per-epoch logs, a checkpoint, held-out
predictions, metrics, hashes, and hardware evidence.

## Ten-line decision contract

```python
from open_jev import Example, Question
from open_jev.model import DynamicDecisionModel
from open_jev.tokenization import WhitespaceTokenizer

question = Question(
    type="choice",
    instructions="Choose the safest tool",
    criteria={
        "search": "read-only web lookup",
        "shell": "local command execution",
        "human": "ask the user first",
    },
)
example = Example(
    id="demo-1",
    state={"request": "Inspect a public documentation page"},
    question=question,
    label="search",
)
tokenizer = WhitespaceTokenizer.fit([example])
model = DynamicDecisionModel(len(tokenizer), hidden_size=64)
```

The same typed contract supports `Choice`, `Noul`, and `Score` questions. JSONL
examples preserve `id`, `group`, `source`, and raw `state` for auditability;
grouped splitting prevents identical states from leaking across evaluation
partitions.

## What it is good for

- Agent route and skill selection
- Tool and action pre-checks
- RAG relevance and evidence checks
- Typed evaluation signals
- Risk triage before deterministic policy enforcement

## Architecture

```text
JSONL -> schema -> grouped split -> tokenizer -> encoder -> dynamic scorer
                                             |-> calibration / audits
                                             |-> benchmark / replay / evidence
                                             |-> checkpoint -> FastAPI
```

The default baseline uses a deterministic whitespace tokenizer, mean-pooled
embeddings, and a permutation-equivariant candidate scorer. The package also
includes provenance receipts, cross-fit temperature calibration, option-order
audits, benchmark reports, direct-logit readout contracts, TorchScript export,
evidence references, and replay keys.

## Reproducible results

The checked-in [synthetic-v2 run](examples/synthetic-v2/README.md) is a pipeline
smoke test, not evidence of real-world or TypeSafe Jev parity. Its 180 held-out
rows reached 100% accuracy after 40 epochs on a synthetic rule-learning task.
Use the runner with representative labeled data before drawing product or model
conclusions. Report accuracy, calibration, latency, and cost separately.

See the [中文训练报告](docs/reports/2026-09-21-h800-synthetic.md) and the
[20-repository integration audit](docs/research/2026-09-22-openjev-20-repo-audit.md).

## What it is not

open-jev is an independent open implementation. It does not reproduce
TypeSafe Jev's private architecture, weights, or training data. It is a modular
research and engineering baseline, not a drop-in safety boundary: deterministic
permissions, sandboxing, human confirmation, and domain-specific validation
remain necessary for consequential actions.

## Development

```bash
ruff check src tests scripts
pytest -q
```

Contributions should include focused tests and a reproducible example when they
add a new decision primitive or evaluation path. Read [Contributing](CONTRIBUTING.md)
and [Citation](CITATION.cff) before publishing results. See the
[20-repository audit](docs/research/2026-09-22-openjev-20-repo-audit.md) for the
source and adaptation record behind the integrated modules.

Licensed under Apache-2.0.
