"""Calibration utilities for typed decision probabilities."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    def transform(self, logits: Tensor) -> Tensor:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        return (logits / self.temperature).softmax(-1)

    @classmethod
    def fit(
        cls, logits: Tensor, target: Tensor, *, steps: int = 100, lr: float = 0.05
    ) -> "TemperatureScaler":
        if logits.ndim != 2 or target.ndim != 2 or logits.shape != target.shape:
            raise ValueError("logits and target must be matching 2D tensors")
        log_t = torch.zeros((), dtype=logits.dtype, device=logits.device, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            opt.zero_grad()
            loss = -(target * (logits / log_t.exp()).log_softmax(-1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return cls(float(log_t.detach().exp().clamp_min(1e-4)))


def nll(probabilities: Tensor, target: Tensor) -> float:
    return float(-(target * probabilities.clamp_min(1e-12).log()).sum(-1).mean())


def brier(probabilities: Tensor, target: Tensor) -> float:
    return float(((probabilities - target) ** 2).sum(-1).mean())


def ece(probabilities: Tensor, target: Tensor, bins: int = 10) -> float:
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(target.argmax(-1)).float()
    result = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        selected = (confidence >= lo) & ((confidence < hi) if i < bins - 1 else (confidence <= hi))
        if selected.any():
            result += float(selected.float().mean()) * abs(
                float(correct[selected].mean() - confidence[selected].mean())
            )
    return result


def report(probabilities: Tensor, target: Tensor) -> dict[str, float]:
    return {
        "nll": nll(probabilities, target),
        "brier": brier(probabilities, target),
        "ece": ece(probabilities, target),
    }
