"""Optional direct next-token readout protocol; no HF import on default path."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Protocol
import torch
from torch import Tensor


class TokenizerProtocol(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...


class CausalModelProtocol(Protocol):
    def __call__(self, input_ids: Tensor, **kwargs: Any) -> Any: ...


def readout_single_token(
    model: CausalModelProtocol,
    tokenizer: TokenizerProtocol,
    prompt: str,
    options: Sequence[str],
    *,
    device: str = "cpu",
) -> dict[str, float]:
    if not options or len(set(options)) != len(options):
        raise ValueError("options must be non-empty and unique")
    token_ids = [list(tokenizer.encode(option, add_special_tokens=False)) for option in options]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("all options must encode to exactly one token")
    input_ids = torch.tensor(
        [list(tokenizer.encode(prompt, add_special_tokens=False))], dtype=torch.long, device=device
    )
    result = model(input_ids)
    logits = result.logits if hasattr(result, "logits") else result[0]
    scores = logits[0, -1, torch.tensor([ids[0] for ids in token_ids], device=logits.device)]
    probs = scores.softmax(-1).tolist()
    return dict(zip(options, (float(p) for p in probs), strict=True))
