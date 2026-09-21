from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
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
    device: str = "auto"


class Trainer:
    def __init__(
        self,
        model: DynamicDecisionModel,
        tokenizer: WhitespaceTokenizer,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config or TrainingConfig()
        requested = self.config.device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        self.model.to(self.device)
        self.selected_epoch = 0

    def _loss(self, batch: Batch) -> torch.Tensor:
        logits = self.model(
            batch.state_tokens.to(self.device), batch.candidate_tokens.to(self.device)
        )
        return masked_soft_cross_entropy(
            logits, batch.target.to(self.device), batch.mask.to(self.device)
        )

    def fit(
        self,
        train: list[Example],
        validation: list[Example] | None = None,
        *,
        on_epoch: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        if not train or self.config.epochs < 1 or self.config.batch_size < 1:
            raise ValueError("training requires data, positive epochs and positive batch_size")
        # The caller seeds before constructing the model; this RNG controls batch order.
        rng = random.Random(self.config.seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        history = []
        best_loss = float("inf")
        best_state = None
        for epoch in range(self.config.epochs):
            self.model.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            total = 0.0
            for start in range(0, len(order), self.config.batch_size):
                selected = [train[i] for i in order[start : start + self.config.batch_size]]
                batch = collate(selected, self.tokenizer)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                loss.backward()
                optimizer.step()
                total += loss.detach().item() * len(selected)
            row = {"epoch": float(epoch + 1), "train_loss": total / len(train)}
            if validation:
                self.model.eval()
                with torch.no_grad():
                    loss_sum = 0.0
                    for start in range(0, len(validation), self.config.batch_size):
                        selected = validation[start : start + self.config.batch_size]
                        loss_sum += self._loss(collate(selected, self.tokenizer)).item() * len(
                            selected
                        )
                    row["validation_loss"] = loss_sum / len(validation)
                if row["validation_loss"] < best_loss:
                    best_loss = row["validation_loss"]
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    self.selected_epoch = epoch + 1
            else:
                self.selected_epoch = epoch + 1
            history.append(row)
            if on_epoch is not None:
                on_epoch(row)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, examples: list[Example]) -> list[dict[str, object]]:
        self.model.eval()
        results = []
        for start in range(0, len(examples), self.config.batch_size):
            selected = examples[start : start + self.config.batch_size]
            batch = collate(selected, self.tokenizer)
            with torch.no_grad():
                probabilities = (
                    self.model(
                        batch.state_tokens.to(self.device), batch.candidate_tokens.to(self.device)
                    )
                    .masked_fill(~batch.mask.to(self.device), -1e9)
                    .softmax(-1)
                    .cpu()
                )
            for i, labels in enumerate(batch.labels):
                best = int(probabilities[i].argmax())
                results.append(
                    {
                        "id": selected[i].id,
                        "type": selected[i].question.type,
                        "label": labels[best],
                        "confidence": float(probabilities[i, best]),
                        "probabilities": dict(
                            zip(labels, probabilities[i, : len(labels)].tolist())
                        ),
                    }
                )
        return results

    def save(self, path: str | Path) -> None:
        """Save inference weights and metadata; optimizer resumption is not supported."""
        torch.save(
            {
                "format_version": 1,
                "model": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
                "vocab": self.tokenizer.vocab,
                "max_length": self.tokenizer.max_length,
                "hidden_size": self.model.encoder.embedding.embedding_dim,
                "config": asdict(self.config),
                "selected_epoch": self.selected_epoch,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, hidden_size: int | None = None, *, device: str = "cpu"
    ) -> "Trainer":
        state = torch.load(path, map_location="cpu", weights_only=True)
        tokenizer = WhitespaceTokenizer(state["vocab"], state["max_length"])
        width = (
            hidden_size
            or state.get("hidden_size")
            or state["model"]["encoder.embedding.weight"].shape[1]
        )
        model = DynamicDecisionModel(len(tokenizer), hidden_size=width)
        model.load_state_dict(state["model"])
        config = TrainingConfig(**{**state.get("config", {}), "device": device})
        trainer = cls(model, tokenizer, config)
        trainer.selected_epoch = state.get("selected_epoch", 0)
        return trainer
