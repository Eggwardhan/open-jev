# Synthetic Dataset and Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic synthetic decision dataset, a runnable training/evaluation script, and persisted artifacts so users can reproduce an actual open-jev training run locally.

**Architecture:** Generate mixed Choice, Noul, and Score examples from a small rule-based simulator with group-aware splits. Train the existing dynamic candidate model through the public `Trainer`, report loss and typed metrics, and save JSONL data, history, predictions, and a checkpoint under an explicit output directory.

**Tech Stack:** Python 3.12, Pydantic, PyTorch, pytest, JSONL.

### Task 1: Define the synthetic data contract with tests

**Files:** `tests/test_synthetic.py`, `src/open_jev/synthetic.py`

Write tests first for deterministic generation, valid typed labels, and group-isolated splits. Run the focused tests to observe the expected missing-module failure, then implement the minimal generator.

### Task 2: Make predictions and metrics use stable typed labels

**Files:** `tests/test_batching.py`, `src/open_jev/batching.py`, `src/open_jev/training.py`

Test that Choice predictions return criterion keys, Score predictions return level values, and mixed candidate counts are masked. Implement label metadata separately from display text.

### Task 3: Add a reproducible training runner

**Files:** `tests/test_train_synthetic.py`, `scripts/train_synthetic.py`, `README.md`

Test the runner on a small fixture and require it to create data, split metadata, history, predictions, and a checkpoint. Run it with a modest dataset and epochs, then document the exact command and expected outputs.

### Task 4: Verify, commit, and publish

Run ruff, the complete pytest suite, and the actual training command. Inspect artifacts and git diff, commit the changes, push to `main`, and verify the remote SHA.

## Execution outcome

- [x] Deterministic v2 data generator with unique cases, shuffled Choice candidates and three tasks.
- [x] Regression tests reproduce and fix unknown Noul candidate tokens and checkpoint architecture mismatch.
- [x] Seeded runner, CUDA support, validation checkpoint selection, train-majority and initialized baselines.
- [x] Actual H800 run: 1,800 rows, 40 epochs, 24.93 seconds, 180/180 test predictions correct.
- [x] Dataset split/source/file hashes checked; GPU weights reloaded locally on CPU.
- [x] Full dataset, 52 KB checkpoint, predictions, plots, and measured report prepared for publication.

Implementation proceeded directly in the clean authorized checkout. A read-only code-review pass flagged the direct Trainer seeding contract; documentation now distinguishes caller-controlled model initialization from config-controlled batch ordering. The synthetic result is explicitly limited to nine risk/latency combinations and unused calibration data.
