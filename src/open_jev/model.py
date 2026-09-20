from __future__ import annotations

import torch
from torch import Tensor, nn


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


class MeanTextEncoder(nn.Module):
    """Small replaceable baseline encoder; a HF encoder can implement the same role."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, tokens: Tensor) -> Tensor:
        mask = tokens.ne(0).unsqueeze(-1)
        values = self.embedding(tokens) * mask
        return values.sum(1) / mask.sum(1).clamp_min(1)


class DynamicDecisionModel(nn.Module):
    """Scores a dynamic candidate axis without fixed label ids."""

    def __init__(self, vocab_size: int, hidden_size: int = 128, max_length: int = 256) -> None:
        super().__init__()
        self.encoder = MeanTextEncoder(vocab_size, hidden_size)
        self.max_length = max_length
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, state_tokens: Tensor, candidate_tokens: Tensor) -> Tensor:
        state = self.encoder(state_tokens)
        batch, choices, length = candidate_tokens.shape
        candidates = self.encoder(candidate_tokens.reshape(batch * choices, length)).reshape(
            batch, choices, -1
        )
        shared = state.unsqueeze(1).expand(-1, choices, -1)
        return self.scorer(torch.cat((shared, candidates), dim=-1)).squeeze(-1)
