# open-jev

[中文文档](README.zh-CN.md)

![open-jev decision graph](docs/assets/open-jev-banner.jpg)

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

## Local model backends

`open-jev` now has a model registry rather than a single-model assumption. The
registry records the model family, the local loading path, and the repository
that motivated the adapter:

| Model family | Local adapter | Models currently registered | Source implementation |
|---|---|---|---|
| Causal next-token readout | `LocalDecisionModel` | Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3-4B-Instruct-2507 | [zhihz/openjev](https://github.com/zhihz/openjev), [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) |
| Encoder candidate logits | `EncoderDecisionModel` | `heman10x/rlcd-modernbert-151m`, `knowledgator/gliclass-modern-base-v2.0` | [openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) |
| Local SGLang/vLLM service | `SGLangDecisionModel` | Qwen3.6-35B-A3B, DiffusionGemma 26B | [openjev-sglang](https://github.com/ekzhang/openjev-sglang), [razorback16/openjev](https://github.com/razorback16/openjev) |

The adapters share the same `Question` contract and return a selected label,
probabilities, confidence, and model identity. The causal adapter performs a
single-token letter readout; the encoder adapter uses the `<<LABEL>>...<<SEP>>`
layout used by the ModernBERT/GLiClass route; the SGLang adapter talks only to a
user-provided local endpoint. None of these paths call TypeSafe's hosted API.

Install the optional dependencies only for the backend you need:

```bash
pip install -e '.[hf]'       # Qwen and other Hugging Face causal models
pip install -e '.[encoder]'  # ModernBERT/GLiClass checkpoints
pip install sglang            # only on the host that will serve a local model
```

If the host cannot reach `huggingface.co`, use a Hugging Face mirror when
downloading weights. The framework still receives a normal local path or model
ID; the mirror is only a download setting:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "knowledgator/gliclass-modern-base-v2.0",
    local_dir="models/gliclass-modern-base-v2.0",
)
PY
```

List the supported model registrations with:

```python
from open_jev.backends import list_backends
for backend in list_backends():
    print(backend.model_id, backend.kind.value, backend.source_repo)
```

Run a local causal model directly:

```bash
open-jev local-decide \
  --model Qwen/Qwen3-0.6B \
  --state 'The customer was charged twice.' \
  --type choice \
  --instructions 'Which queue should handle this?' \
  --criteria '{"billing":"payments and refunds","technical":"software bugs"}' \
  --device cpu
```

Load an encoder checkpoint with the same decision contract:

```python
from open_jev.encoder_model import EncoderDecisionModel
from open_jev.schema import Question

model = EncoderDecisionModel.from_pretrained(
    "knowledgator/gliclass-modern-base-v2.0",
    device="cuda",  # use "cpu" for an offline smoke test
)
question = Question(
    type="choice",
    instructions="Which queue should handle this?",
    criteria={"billing": "payments and refunds", "technical": "software bugs"},
)
print(model.decide({"customer_message": "charged twice"}, question))
```

For an offline mirror download, pass the downloaded directory instead of the
Hub ID:

```python
model = EncoderDecisionModel.from_pretrained(
    "models/gliclass-modern-base-v2.0", device="cpu"
)
```

Start a local SGLang service on the model host. Vanilla SGLang exposes its
standard generation endpoints; the service used by this adapter must additionally
expose the Jev-compatible `/v1/systemone` endpoint. The
`openjev-sglang` implementation is one reference for that wrapper:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.25
```

Then call the service through the generic registry:

```python
from open_jev.backends import load_backend
from open_jev.schema import Question

model = load_backend(
    "Qwen/Qwen3-0.6B",
    kind="sglang",
    endpoint="http://127.0.0.1:30000",
)
question = Question(type="noul", instructions="Is the system ready?")
print(model.decide({"status": "ready"}, question))
```

For larger models such as Qwen3.6-35B-A3B, replace `--model-path` and ensure
the GPU has enough free memory before starting the service. The SGLang adapter
does not download weights and does not call a hosted API; it only sends the
typed decision request to the endpoint you provide.

The Qwen model IDs above come from the public model cards and local inference
instructions for [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Their
reported model availability and license metadata should be rechecked before
redistributing downloaded weights.

The built-in registry is a provenance catalog, not an allowlist. A user model can
be loaded without changing framework code by declaring its backend kind:

```python
from open_jev.backends import load_backend

backend = load_backend(
    "org/my-causal-model",
    kind="causal",       # causal, encoder, or sglang
    device="cuda",
)
result = backend.decide(state, question)
```

Projects can also register a model for discoverability with
`register_backend(BackendSpec(...))`. The core decision contract does not depend
on Qwen, ModernBERT, GLiClass, DiffusionGemma, or any specific repository.
