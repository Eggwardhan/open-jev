from __future__ import annotations

from torch import Tensor


def masked_soft_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
    return -(target * logp).sum(-1).mean()


def brier_score(probabilities: Tensor, target: Tensor, mask: Tensor) -> float:
    return float((((probabilities - target) ** 2) * mask).sum(-1).mean().item())


def accuracy(probabilities: Tensor, target: Tensor) -> float:
    return float((probabilities.argmax(-1) == target.argmax(-1)).float().mean().item())


def expected_calibration_error(probabilities: Tensor, target: Tensor, bins: int = 10) -> float:
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(target.argmax(-1)).float()
    result = 0.0
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        selected = (confidence >= lower) & (
            confidence < upper if i < bins - 1 else confidence <= upper
        )
        if selected.any():
            result += float(selected.float().mean()) * abs(
                float(correct[selected].mean() - confidence[selected].mean())
            )
    return result
