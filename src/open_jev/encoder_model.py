"""Local encoder-style decision backend for ModernBERT/GLiClass checkpoints."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import json
import torch
from .schema import Question


def _options(question: Question) -> list[tuple[str, str]]:
    if question.type == "choice":
        assert isinstance(question.criteria, dict)
        return list(question.criteria.items())
    if question.type == "score":
        assert isinstance(question.criteria, list)
        return [(str(i), value) for i, value in enumerate(question.criteria)]
    return [("false", "The proposition is false"), ("true", "The proposition is true")]


def _state_text(state: str | Mapping[str, Any] | Sequence[Any]) -> str:
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )


def build_encoder_prompt(state: str | Mapping[str, Any] | Sequence[Any], question: Question) -> str:
    """Build the label-marker layout used by the Verdict/GLiClass route."""
    labels = [description for _, description in _options(question)]
    label_prefix = "".join(f"<<LABEL>>{label}" for label in labels)
    return (
        f"{label_prefix}<<SEP>>Question: {question.instructions}\n\nContext:\n{_state_text(state)}"
    )


class EncoderDecisionModel:
    """Adapt an encoder/GLiClass model exposing candidate logits."""

    def __init__(
        self, model: Any, tokenizer: Any, *, model_id: str | None = None, device: str = "cpu"
    ):
        self.model, self.tokenizer = model, tokenizer
        self.model_id, self.device = model_id, device

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, device: str = "auto", revision: str | None = None
    ) -> "EncoderDecisionModel":
        try:
            from transformers import AutoTokenizer
            from gliclass import GLiClassModel
        except ImportError as exc:
            raise RuntimeError(
                "install the optional encoder extra: pip install -e '.[encoder]'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = GLiClassModel.from_pretrained(model_id)
        if device != "auto":
            model.to(device)
        model.eval()
        return cls(model, tokenizer, model_id=model_id, device=device)

    def decide(
        self, state: str | Mapping[str, Any] | Sequence[Any], question: Question
    ) -> dict[str, Any]:
        options = _options(question)
        texts = [
            f"Context: {_state_text(state)}\nQuestion: {question.instructions}\nCandidate: {description}"
            for _, description in options
        ]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if hasattr(encoded, "to") and self.device != "auto":
            encoded = encoded.to(self.device)
        elif isinstance(encoded, dict) and self.device != "auto":
            encoded = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded)
        logits = output.logits if hasattr(output, "logits") else output[0]
        logits = logits.float()
        if logits.ndim == 2 and logits.shape[0] == len(options):
            scores = logits[:, -1] if logits.shape[1] > 1 else logits[:, 0]
        elif logits.ndim == 2 and logits.shape[0] == 1:
            scores = logits[0, : len(options)]
        else:
            raise ValueError("encoder output must be [candidates, classes] or [1, candidates]")
        probs = scores.softmax(-1).tolist()
        probabilities = {key: float(p) for (key, _), p in zip(options, probs, strict=True)}
        label = max(probabilities, key=probabilities.get)
        result: dict[str, Any] = {
            "label": label,
            "probabilities": probabilities,
            "confidence": probabilities[label],
            "model": self.model_id,
        }
        if question.type == "score":
            result["score"] = sum(i * probabilities[key] for i, (key, _) in enumerate(options))
        if question.type == "noul":
            result["probability"] = probabilities["true"]
        return result
