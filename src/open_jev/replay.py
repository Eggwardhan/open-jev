"""Deterministic cache keys and replay receipts for shared-state decisions."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any


def replay_key(state: Any, question: Any, *, version: str = "1") -> str:
    payload = json.dumps(
        {"version": version, "state": state, "question": question},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ReplayReceipt:
    key: str
    state: Any
    question: Any
    output: Any

    @classmethod
    def create(cls, state: Any, question: Any, output: Any) -> "ReplayReceipt":
        return cls(replay_key(state, question), state, question, output)

    def verify(self) -> bool:
        return self.key == replay_key(self.state, self.question)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
