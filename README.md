# open-jev

An independent, modular PyTorch framework for typed decision models. It turns a state plus a typed question into a probability distribution over dynamic candidates, with replaceable tokenizers, encoders, scoring heads, training loops, metrics, and HTTP serving.

`open-jev` is **not** TypeSafe's Jev model and does not claim to reproduce its private architecture, weights, or training data. It is an open implementation inspired by the public problem shape: calibrated decisions over Choice, Noul, and Score questions. See the [TypeSafe introduction](https://docs.typesafe.ai/introduction) for the hosted product.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Run an actual training experiment

```bash
open-jev train-synthetic --output artifacts/my-run \
  --groups 600 --epochs 40 --batch-size 32 --seed 7 --device auto
```

Use `--device cuda` to require CUDA (fails if unavailable), or `--device cpu`.
The source-checkout equivalent is `PYTHONPATH=src python scripts/train_synthetic.py`
with the same arguments. The runner creates four JSONL splits, per-epoch logs,
a self-describing checkpoint, held-out predictions and metrics, hashes, and
hardware evidence. It restores the best validation checkpoint before evaluation.

A complete [recorded H800 run](examples/synthetic-v2/README.md), including data
and weights, is checked in. Across 180 held-out rows, the initialized model
achieved **41.7%** accuracy, a train-majority baseline **52.2%**, and the trained
model **100%** after 40 epochs (9.6 seconds). This is a simple synthetic rule
learning demonstration with 12,225 parameters, not evidence of real-world or
TypeSafe Jev parity. The reserved calibration split is unused.

![Actual H800 training curves](examples/synthetic-v2/curves.png)

[中文训练报告](docs/reports/2026-09-21-h800-synthetic.md)

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

This repository is a clean, runnable baseline rather than a proprietary-model reproduction. The included synthetic checkpoint is only a pipeline demonstration. No pretrained language model or TypeSafe parity is provided, and domain calibration requires representative held-out data. Before production use, add task-specific data governance, adversarial tests, monitoring, access control, and temperature calibration.

## Development

```bash
ruff check src tests scripts
pytest -q
```

Licensed under Apache-2.0.

## Integration provenance: 20 related repositories

The table below records the design source for each integrated or explicitly
rejected idea. The implementation in this repository is original and does not
copy private weights, training data, or source code from these projects.

| Source repository | Design reviewed | Local result |
|---|---|---|
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | frozen splits, calibration and experiment receipts | `provenance.py`, `calibration.py`; adapted |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) | direct option-logit readout and shared/separate comparisons | `readout.py`; adapted as an optional protocol |
| [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | versioned prompts and typed response validation | schema and readout contracts; adapted |
| [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | frozen feature metadata and context-shuffle control | `audits.py`; adapted |
| [wfzyx/von](https://github.com/wfzyx/von) | backend-independent candidate axis | existing dynamic scorer; retained |
| [Heman10x-NGU/openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) | permutation robustness and calibration artifacts | `audits.py`, `calibration.py`; adapted; weights not copied |
| [ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) | packed prompts and temperature scaling | `calibration.py`; adapted |
| [razorback16/openjev](https://github.com/razorback16/openjev) | typed service and backend separation | `serve.py` boundary; reference only |
| [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) | latency, usage and branch-level failure reporting | `benchmark.py`; adapted |
| [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) | separate accuracy, calibration, speed and cost axes | `benchmark.py`; adapted |
| [receptron/laya](https://github.com/receptron/laya) | export and sequence validation contracts | `export.py`; adapted; ONNX remains optional |
| [nico-martin/open-jev](https://github.com/nico-martin/open-jev) | typed question and unique-option validation | schema/readout validation; adapted |
| [kyegomez/open-jev](https://github.com/kyegomez/open-jev) | shared state encoder and typed heads | existing model architecture; design reference |
| [intikhab49/open-jev-typed-decision-engine](https://github.com/intikhab49/open-jev-typed-decision-engine) | separated train/calibrate/evaluate/export stages | runner plus `calibration.py`/`export.py`; adapted |
| [kshetrajna12/reflex](https://github.com/kshetrajna12/reflex) | option-order and packed-inference audits | `audits.py`; adapted |
| [deepanwadhwa/OpenDecision](https://github.com/deepanwadhwa/OpenDecision) | evidence references alongside decisions | `evidence.py`; adapted |
| [IamBusy/OpenJev-Vision](https://github.com/IamBusy/OpenJev-Vision) | branch-cache keys and replayable experiments | `replay.py`; adapted without image dependency |
| [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev) | prompt parity and shared-input identity | provenance/replay keys; adapted |
| [mithalouni/system-one-open](https://github.com/mithalouni/system-one-open) | separated training/evaluation reports | `benchmark.py`; adapted |
| [logicrw/awesome-jev-projects](https://github.com/logicrw/awesome-jev-projects) | source review, receipts and exclusion records | this section and `docs/research/2026-09-22-openjev-20-repo-audit.md` |

For the code-level review, license notes, and keep/adapt/reject decisions, see
[`docs/research/2026-09-22-openjev-20-repo-audit.md`](docs/research/2026-09-22-openjev-20-repo-audit.md).

### Sequential integration commits

The additions were landed independently so each source-derived capability can be
reviewed or reverted without mixing model code and documentation:

- `e680606` — provenance receipts and source/environment hashes
- `e2a5c88` — calibration, robustness audits, benchmark contract, direct-logit readout
- `759f9c9` — TorchScript export, evidence references, replay receipts
- `826418c` — replay/evidence metadata in `/v1/decide`
- `f30240c` — cross-fit calibration and named baseline comparison
