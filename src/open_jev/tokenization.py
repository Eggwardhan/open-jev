from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .schema import Example


def text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass
class WhitespaceTokenizer:
    vocab: dict[str, int]
    max_length: int = 128

    @classmethod
    def fit(
        cls, examples: Iterable[Example], max_vocab: int = 20_000, max_length: int = 128
    ) -> "WhitespaceTokenizer":
        counts: Counter[str] = Counter()
        for item in examples:
            counts.update(cls._tokens(cls._example_text(item)))
        vocab = {"<pad>": 0, "<unk>": 1}
        for token, _ in counts.most_common(max_vocab - 2):
            vocab.setdefault(token, len(vocab))
        return cls(vocab, max_length)

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", value.lower(), flags=re.UNICODE)

    @staticmethod
    def _example_text(item: Example) -> str:
        return " ".join(
            (
                text(item.state),
                item.question.instructions,
                "false true" if item.question.type == "noul" else text(item.question.criteria),
            )
        )

    def encode(self, value: object) -> list[int]:
        ids = [self.vocab.get(token, 1) for token in self._tokens(text(value))][: self.max_length]
        return ids or [1]

    def __len__(self) -> int:
        return len(self.vocab)
