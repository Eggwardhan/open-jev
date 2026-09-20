# Open Jev implementation plan

**Goal:** Build and publish an independently implemented, modular decision-model framework with real data preparation, training, calibration, evaluation and serving.

**Architecture:** Typed examples unify Choice, Noul and Score as distributions over dynamic natural-language candidates. Replaceable tokenizers and backbones encode query and candidates; replaceable heads score candidates. Training objectives, data adapters, calibration and metrics remain independent. PyTorch performs actual gradient training; Transformers supplies optional pretrained backbones.

**Tech stack:** Python 3.12, PyTorch, Transformers, Pydantic, FastAPI, pytest, uv.

## Requirements and evidence

- [ ] Strict data and inference contracts, arbitrary candidate sets, soft/hard labels; invalid examples rejected.
- [ ] JSONL ingestion, provenance, deduplication, group-isolated train/validation/calibration/test splits, hashes and manifests.
- [ ] Offline tiny Transformer and pretrained Hugging Face backbone; swappable heads and tokenizer; no label-index lookup shortcut.
- [ ] Cross entropy, Brier and soft-target distillation objectives; training config, seed, device, batching, clipping, validation selection, checkpoint/resume.
- [ ] Temperature calibration on separate split; probability metrics, accuracy/F1, score error, selective risk, latency and per-example outputs.
- [ ] Choice/Noul/Score prediction and HTTP endpoint from trained artifacts; malformed requests fail explicitly.
- [ ] Real local train -> reload -> calibrate -> evaluate -> predict -> HTTP test, and pretrained backend integration verification.
- [ ] English/Chinese documentation, data format, extension guide, model card, research limitations, example configs, CI and license.
- [ ] Public GitHub repository, narrow commit, remote SHA verification and CI result.

## Approach choices

1. **Dynamic bi-encoder plus interaction head (default):** reuse state/question encoding, batch candidate encodings, preserve candidate permutation equivariance. Efficient and transparent but less expressive than full cross attention.
2. **Cross-encoder scorer:** score each state/question/candidate together; higher interaction capacity, greater repeated input cost. Add as replaceable architecture with shared training contracts.
3. **Generative token-logit adapter:** useful baseline but sensitive to tokenization and candidate names; not the default learned model.

No proprietary architecture replication or frontier performance is claimed. Tiny synthetic runs prove pipeline learning and artifact integrity, not semantic generalization. Pretrained experiments and real datasets must have separately recorded results.

## Implementation sequence

1. Tests for schema, target distributions, data splits and leakage; implement `schema.py`, `data.py` and adapters.
2. Tests for ragged candidates, padding, permutation equivariance and gradients; implement `model.py`, tokenization and registries.
3. Tests for objective values, calibration, metrics and validation-only model selection; implement training/evaluation modules.
4. End-to-end tests for checkpoint reload, commands and API; implement CLI and serving.
5. Run offline learned example and a small pretrained integration. Record failures and limitations; do not relabel fixtures as model benchmarks.
6. Review artifacts, packaging and secrets; publish public repo and verify remote/CI.

## Data and evaluation decisions

Keep source/group identifiers and licenses on examples. Group isolation includes identical state content even when ids differ. Fit vocabulary only on training. Use validation for checkpoint selection, calibration for temperature/threshold selection, test solely for final metrics. Preserve raw predictions and failed outcomes. Report uncertainty and class support; no ECE-only claims of safety.

## Publication

User authorized public `Eggwardhan/open-jev` on 2026-09-20. Original code under Apache-2.0; dependencies and external datasets retain their licenses. Never commit credentials, local corpora, runtime caches or checkpoints by default.
