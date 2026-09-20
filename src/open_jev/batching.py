from __future__ import annotations

from dataclasses import dataclass
import torch

from .schema import Example
from .tokenization import WhitespaceTokenizer, text


def candidates(example: Example) -> list[str]:
    q = example.question
    if q.type == "choice":
        return [f"{key}: {value}" for key, value in q.criteria.items()]
    if q.type == "noul":
        return ["false", "true"]
    return [text(level) for level in q.criteria]


@dataclass
class Batch:
    state_tokens: torch.Tensor
    candidate_tokens: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    labels: list[list[str]]
    examples: list[Example]


def collate(examples: list[Example], tokenizer: WhitespaceTokenizer) -> Batch:
    encoded_states = [
        tokenizer.encode(f"{text(e.state)} {e.question.instructions}") for e in examples
    ]
    encoded_candidates = [[tokenizer.encode(c) for c in candidates(e)] for e in examples]
    max_state = max(map(len, encoded_states))
    max_choices = max(map(len, encoded_candidates))
    max_len = tokenizer.max_length
    state = torch.zeros((len(examples), max_state), dtype=torch.long)
    cand = torch.zeros((len(examples), max_choices, max_len), dtype=torch.long)
    target = torch.zeros((len(examples), max_choices), dtype=torch.float32)
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i, e in enumerate(examples):
        state[i, : len(encoded_states[i])] = torch.tensor(encoded_states[i])
        for j, ids in enumerate(encoded_candidates[i]):
            cand[i, j, : len(ids)] = torch.tensor(ids)
        target[i, : len(e.target())] = torch.tensor(e.target())
        mask[i, : len(encoded_candidates[i])] = True
    return Batch(
        state, cand, target, mask, [[c for c in candidates(e)] for e in examples], examples
    )
