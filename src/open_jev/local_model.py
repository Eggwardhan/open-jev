"""Local Hugging Face causal-model backend for Jev-style decisions.

The backend reads one next-token distribution over letter labels. It never calls
TypeSafe or another hosted decision API. Transformers is imported only by
``from_pretrained`` so the base package remains lightweight.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .readout import readout_single_token
from .schema import Question

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_SMALL_MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B")


def _state_text(state: str | Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _options(question: Question) -> list[tuple[str, str]]:
    if question.type == "choice":
        assert isinstance(question.criteria, dict)
        return list(question.criteria.items())
    if question.type == "score":
        assert isinstance(question.criteria, list)
        return [(str(index), value) for index, value in enumerate(question.criteria)]
    return [("false", "The proposition is false"), ("true", "The proposition is true")]


def build_decision_prompt(
    state: str | Mapping[str, Any] | Sequence[Any], question: Question
) -> str:
    """Build an explicit, closed-set prompt for local causal models."""
    options = _options(question)
    lines = [
        "You are a decision classifier.",
        "Read the context and question, then return exactly one label.",
        f"Context: {_state_text(state)}",
        f"Question: {question.instructions}",
        "Options:",
    ]
    for index, (_, description) in enumerate(options):
        lines.append(f"{chr(65 + index)}: {description}")
    lines.append(
        "Return exactly one label from: " + ", ".join(chr(65 + i) for i in range(len(options)))
    )
    return "\n".join(lines)


class LocalDecisionModel:
    """Run typed decisions with a local causal language model."""

    def __init__(
        self, model: Any, tokenizer: Any, *, model_id: str | None = None, device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_LOCAL_MODEL,
        *,
        revision: str | None = None,
        device: str = "auto",
        torch_dtype: Any = "auto",
    ) -> "LocalDecisionModel":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the optional HF extra: pip install -e '.[hf]'") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": torch_dtype}
        if revision is None:
            kwargs.pop("revision")
        if device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device != "auto":
            model.to(device)
        model.eval()
        return cls(model, tokenizer, model_id=model_id, device=device)

    def _render_prompt(
        self, state: str | Mapping[str, Any] | Sequence[Any], question: Question
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": build_decision_prompt(state, question)}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return build_decision_prompt(state, question)

    def _labels(self, question: Question) -> tuple[list[str], list[str]]:
        options = _options(question)
        labels = [chr(65 + i) for i in range(len(options))]
        keys = [key for key, _ in options]
        return labels, keys

    def decide(
        self, state: str | Mapping[str, Any] | Sequence[Any], question: Question
    ) -> dict[str, Any]:
        labels, keys = self._labels(question)
        prompt = self._render_prompt(state, question)
        import torch

        readout_device = self.device
        if readout_device == "auto":
            try:
                readout_device = str(next(self.model.parameters()).device)
            except (AttributeError, StopIteration):
                readout_device = "cpu"
        with torch.inference_mode():
            label_probabilities = readout_single_token(
                self.model, self.tokenizer, prompt, labels, device=readout_device
            )
        probabilities = {
            key: label_probabilities[label] for key, label in zip(keys, labels, strict=True)
        }
        selected_key = max(probabilities, key=probabilities.get)
        result: dict[str, Any] = {
            "label": selected_key,
            "probabilities": probabilities,
            "confidence": max(probabilities.values()),
            "model": self.model_id,
        }
        if question.type == "score":
            result["score"] = sum(index * probabilities[key] for index, key in enumerate(keys))
        if question.type == "noul":
            result["probability"] = probabilities["true"]
        return result
