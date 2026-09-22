"""Robustness audits for typed decision outputs."""

from __future__ import annotations
from collections.abc import Callable
import torch
from torch import Tensor


def option_permutation_audit(
    predict: Callable[[Tensor], Tensor], logits: Tensor, permutations: list[Tensor]
) -> dict[str, float | int]:
    base = predict(logits).softmax(-1)
    flips = 0
    drift = 0.0
    for perm in permutations:
        restored = (
            predict(logits.index_select(-1, perm)).softmax(-1).index_select(-1, torch.argsort(perm))
        )
        flips += int(restored.argmax(-1).ne(base.argmax(-1)).any())
        drift += float((restored - base).abs().mean())
    count = len(permutations)
    return {
        "count": count,
        "flip_rate": flips / count if count else 0.0,
        "mean_probability_drift": drift / count if count else 0.0,
    }


def context_shuffle_audit(
    predict: Callable[[Tensor], Tensor], original: Tensor, shuffled: Tensor
) -> dict[str, float | int]:
    a, b = predict(original).softmax(-1), predict(shuffled).softmax(-1)
    return {
        "count": int(a.shape[0]),
        "flip_rate": float(a.argmax(-1).ne(b.argmax(-1)).float().mean()),
        "mean_probability_drift": float((a - b).abs().mean()),
    }
