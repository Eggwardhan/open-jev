# open-jev

An independent, modular PyTorch framework for typed decision models. It turns a state plus a typed question into a probability distribution over dynamic candidates, with replaceable tokenizers, encoders, scoring heads, training loops, metrics, and HTTP serving.

`open-jev` is **not** TypeSafe's Jev model and does not claim to reproduce its private architecture, weights, or training data. It is an open implementation inspired by the public problem shape: calibrated decisions over Choice, Noul, and Score questions. See the [TypeSafe introduction](https://docs.typesafe.ai/introduction) for the hosted product.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The core API is deliberately small:

```python
from open_jev import Example, Question
from open_jev.batching import collate
from open_jev.tokenization import WhitespaceTokenizer
from open_jev.model import DynamicDecisionModel

question = Question(type="choice", instructions="Choose a route", criteria={"safe": "low risk", "fast": "low latency"})
example = Example(id="1", state={"latency_ms": 120}, question=question, label="safe")
tokenizer = WhitespaceTokenizer.fit([example])
model = DynamicDecisionModel(len(tokenizer), hidden_size=64)
```

JSONL examples are validated with Pydantic and preserve `id`, `group`, `source`, and raw `state` for auditability. Splitting isolates groups and identical states before training. Calibration data is kept separate from model fitting.

## Architecture

```text
JSONL -> schema -> grouped split -> tokenizer -> encoder -> dynamic scorer
                                             |-> soft CE / Brier / ECE
                                             |-> checkpoint -> FastAPI
```

The included baseline uses a deterministic whitespace tokenizer, mean pooled embeddings, and a permutation-equivariant candidate scorer. Each part can be replaced without changing the typed data contract. A future `hf` extra is reserved for Hugging Face tokenizers and backbones; the base package does not download model weights.

## Scope and limitations

This repository is a clean, runnable baseline rather than a proprietary-model reproduction. It does not include a pretrained checkpoint, claim TypeSafe parity, or promise domain calibration without a calibration set. Before production use, add task-specific data governance, adversarial tests, monitoring, access control, and temperature calibration.

## Development

```bash
ruff check src tests
pytest -q
```

Licensed under Apache-2.0.

