"""Optional evidence references attached to decisions."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class Evidence:
    source: str
    ref: str
    quote: str | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def merge_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str | None]] = set()
    result = []
    for item in items:
        key = (item.source, item.ref, item.quote)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
