from __future__ import annotations

from dataclasses import dataclass
import random
import torch

from .batching import Batch, collate
from .metrics import masked_soft_cross_entropy
from .model import DynamicDecisionModel
from .schema import Example
from .tokenization import WhitespaceTokenizer


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 3e-3
    batch_size: int = 8
    seed: int = 7


class Trainer:
    def __init__(
        self,
        model: DynamicDecisionModel,
        tokenizer: WhitespaceTokenizer,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config or TrainingConfig()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _loss(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch.state_tokens, batch.candidate_tokens)
        return masked_soft_cross_entropy(logits, batch.target, batch.mask)

    def fit(
        self, train: list[Example], validation: list[Example] | None = None
    ) -> list[dict[str, float]]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, float]] = []
        for epoch in range(self.config.epochs):
            self.model.train()
            order = list(range(len(train)))
            random.shuffle(order)
            total = 0.0
            for start in range(0, len(order), self.config.batch_size):
                batch = collate(
                    [train[i] for i in order[start : start + self.config.batch_size]],
                    self.tokenizer,
                )
                optimizer.zero_grad()
                loss = self._loss(batch)
                loss.backward()
                optimizer.step()
                total += float(loss)
            row = {
                "epoch": float(epoch + 1),
                "train_loss": total
                / max(1, (len(train) + self.config.batch_size - 1) // self.config.batch_size),
            }
            if validation:
                self.model.eval()
                with torch.no_grad():
                    row["validation_loss"] = float(self._loss(collate(validation, self.tokenizer)))
            history.append(row)
        return history

    def predict(self, examples: list[Example]) -> list[dict[str, object]]:
        self.model.eval()
        batch = collate(examples, self.tokenizer)
        with torch.no_grad():
            probabilities = (
                self.model(batch.state_tokens, batch.candidate_tokens)
                .masked_fill(~batch.mask, -1e9)
                .softmax(-1)
            )
        results = []
        for i, labels in enumerate(batch.labels):
            best = int(probabilities[i].argmax())
            results.append(
                {
                    "id": examples[i].id,
                    "type": examples[i].question.type,
                    "label": labels[best],
                    "confidence": float(probabilities[i, best]),
                    "probabilities": dict(zip(labels, probabilities[i, : len(labels)].tolist())),
                }
            )
        return results

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "vocab": self.tokenizer.vocab,
                "max_length": self.tokenizer.max_length,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, hidden_size: int = 64) -> "Trainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        tokenizer = WhitespaceTokenizer(state["vocab"], state["max_length"])
        model = DynamicDecisionModel(len(tokenizer), hidden_size=hidden_size)
        model.load_state_dict(state["model"])
        return cls(model, tokenizer)
