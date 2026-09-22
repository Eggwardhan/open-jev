"""Optional model export contracts; heavyweight runtimes remain optional."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import torch
from torch import Tensor


def export_torchscript(
    model: torch.nn.Module, example_inputs: Sequence[Tensor], path: str | Path
) -> Path:
    model.eval()
    traced = torch.jit.trace(model, tuple(example_inputs), strict=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(target))
    return target


def verify_torchscript(
    path: str | Path, model: torch.nn.Module, example_inputs: Sequence[Tensor]
) -> float:
    reference = model(*example_inputs)
    restored = torch.jit.load(str(path))(*example_inputs)
    if isinstance(reference, (tuple, list)):
        return max(float((a - b).abs().max()) for a, b in zip(reference, restored, strict=True))
    return float((reference - restored).abs().max())
