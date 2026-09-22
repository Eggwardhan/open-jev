# OpenJev 20-Repository Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 20 个同类仓库中可验证且与本项目兼容的训练、推理、校准、审计、评测和导出能力逐项整合到 `open-jev`，并保留来源、边界和验证证据。

**Architecture:** 以当前 `Example`/`Trainer`/`runner` 为稳定核心，新增小而独立的 provenance、calibration、audit、benchmark、readout、export、evidence/replay 模块。可选依赖通过运行时导入隔离；默认安装和 synthetic v2 训练不受影响。

**Tech Stack:** Python 3.12+, PyTorch, Pydantic, NumPy, pytest；可选 Transformers、safetensors、ONNX。

### Task 1: Provenance and research receipts

**Files:** Create `src/open_jev/provenance.py`, `tests/test_provenance.py`; modify `src/open_jev/runner.py` and `docs/research/2026-09-22-openjev-20-repo-audit.md`.

Implement deterministic file/dataset manifests, repository source records, environment capture, and a receipt validator. Keep generated receipts JSON-serializable and fail on missing or changed files.

### Task 2: Calibration

**Files:** Create `src/open_jev/calibration.py`, `tests/test_calibration.py`; modify `src/open_jev/metrics.py` and `src/open_jev/runner.py`.

Implement temperature scaling on logits, cross-fit fitting, NLL/Brier/ECE reporting, and explicit train/calibration/test separation. Do not fit on test rows.

### Task 3: Robustness audits

**Files:** Create `src/open_jev/audits.py`, `tests/test_audits.py`; modify `src/open_jev/runner.py`.

Implement option permutation restoration, semantic-label comparison, paired context-shuffle controls, and packed-vs-separate result summaries. Audits must report flip rate, probability drift, and sample counts.

### Task 4: Unified benchmark report

**Files:** Create `src/open_jev/benchmark.py`, `tests/test_benchmark.py`; modify `src/open_jev/cli.py` and `README.md`.

Expose per-type accuracy, NLL, Brier, ECE, latency and baseline comparison without collapsing them into one score. Save raw predictions and a machine-readable report.

### Task 5: Optional direct-logit readout

**Files:** Create `src/open_jev/readout.py`, `tests/test_readout.py`; modify `pyproject.toml` with an optional HF extra.

Define a tokenizer/model protocol and a safe next-token label readout for choice/noul/score. Enforce single-token labels and declared-option support. No Transformers import on the default path.

### Task 6: Export and evidence/replay contracts

**Files:** Create `src/open_jev/export.py`, `src/open_jev/evidence.py`, `src/open_jev/replay.py`, tests for each; modify `src/open_jev/serve.py` only for additive fields.

Add optional TorchScript/ONNX export parity checks, evidence references, shared-state cache keys, and replay receipts. Keep image support and heavyweight runtimes optional.

### Task 7: End-to-end verification and publication

Run the full unit suite, ruff, synthetic CPU run, artifact verification, and optional export/readout tests. Update the integration report with actual files and test evidence, commit each task separately, push `main`, and verify GitHub CI.

