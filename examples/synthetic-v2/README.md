# Synthetic v2 dataset and H800 run

This directory contains an actual from-scratch training run, including all four
JSONL splits, a small inference checkpoint, predictions, and measured results.
The model is a 12,225-parameter embedding + MLP baseline, not a pretrained LLM.

## Dataset card

- Generator: `src/open_jev/synthetic.py`, seed `7`.
- Cases: 600 distinct states sampled without replacement from 972 combinations.
- Each case produces three questions: Choice, Noul, and Score (1,800 rows).
- State fields: risk band, latency band, case kind, channel, region, service.
- Label rules use risk and latency only. The remaining four fields are nuisance
  features. There are only **nine distinct risk/latency combinations**; increasing
  the number of nuisance combinations does not make this a difficult benchmark.
- Choice: `automate` iff risk is low and latency is not slow; otherwise `review`.
- Noul: `true` iff risk is high or latency is slow (requires escalation).
- Score: low risk → 0; medium → 1; high and not slow → 2; high and slow → 3.
- Choice candidate order varies per case. Score levels keep their ordinal order.
- Labels and case IDs are not included in model input. The tokenizer is fitted
  only on training inputs; no label values are used to build the vocabulary.
- The state is shared by all three questions about a case. Cases and exact states
  do not overlap between splits. Counts: train 1,260; validation 180;
  calibration 180; test 180. Each test task has 60 examples.
- Calibration data is reserved and **unused**. Reported probabilities and ECE
  are uncalibrated; no temperature scaling has been performed.

The rules are intentionally simple and noiseless. A handwritten implementation
of these rules would also be 100% accurate. The test set measures learning the
same rules on unseen state combinations, not real-world language understanding,
robustness, or generalization to new rules, schemas, languages, or tasks.

## Reproduce

From the repository root, after installing the project and a compatible CUDA
PyTorch build:

```bash
open-jev train-synthetic --output artifacts/h800-reproduction \
  --groups 600 --epochs 40 --batch-size 32 --seed 7 --device cuda
```

For CPU, replace `--device cuda` with `--device cpu`. Use a new/empty output
folder for every run. Reproducibility means deterministic data generation and
seeded initialization/batch order. Exact GPU bits are not promised across
hardware or library versions. The runner seeds PyTorch **before** constructing
the model. When using `Trainer` directly, call `torch.manual_seed(seed)` before
`DynamicDecisionModel(...)`; `TrainingConfig.seed` controls batch order and
does not reset an already constructed model or its weights.

The shared environment used for the recorded run was Python 3.13.9,
PyTorch 2.7.1+cu128, Pydantic 2.11.7, NumPy 2.2.6, NVIDIA H800 with driver
550.90.07. The network download attempt was not used for training.

## Artifacts

| File | Contents |
|---|---|
| `train.jsonl`, `validation.jsonl`, `calibration.jsonl`, `test.jsonl` | Complete generated dataset |
| `manifest.json` | Configuration, dataset/split/source hashes, environment |
| `history.json`, `history.jsonl`, `training.log` | Per-epoch train and validation loss |
| `metrics.json` | Untrained and train-majority baselines, held-out metrics, device evidence |
| `predictions.json` | Per-example predictions, probabilities and target labels |
| `model.pt` | Small checkpoint with vocabulary, architecture and best validation weights |
| `checksums.json` | SHA-256 of the original training output files |
| `verification.json` | Transfer/source/split checks and local CPU reload comparison |
| `curves.png` | Loss curves and held-out accuracy comparison |

Checkpoint selection uses minimum validation NLL among 40 epochs. The runner
reports the initialized model and selected checkpoint on the same test split;
these results do not select hyperparameters. Train-majority baseline labels are
computed per task using training data only. Brier is the sum of squared errors
across classes, averaged over rows. ECE uses 10 equally spaced confidence bins.
Score accuracy uses argmax; Score MAE uses the expected level on the 0–3 scale.

`source_sha256` hashes the ordered names and bytes of all `src/open_jev/*.py`
files. The remote source was synchronized without `.git`, so `git_head` is null.
Checksums exclude this README, plot, copied console log, and subsequent local
verification, which were added after the training run.

Load the checkpoint on CPU:

```python
from open_jev.data import read_jsonl
from open_jev.training import Trainer

trainer = Trainer.load("examples/synthetic-v2/model.pt", device="cpu")
examples = read_jsonl("examples/synthetic-v2/test.jsonl")
print(trainer.predict(examples[:3]))
```

`predict` returns Choice keys, Noul strings `false`/`true`, and Score level names
as `label`; `probabilities` uses the same stable keys. Training labels for Score
remain numeric levels. Inference weights can be reloaded; optimizer-state resume
is not implemented.
