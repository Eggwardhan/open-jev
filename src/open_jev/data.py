from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Example


def canonical_state(state: object) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: str | Path) -> list[Example]:
    rows: list[Example] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(Example.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid example at {path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def write_jsonl(rows: Iterable[Example], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(row.model_dump_json() for row in rows) + "\n")


def dataset_fingerprint(rows: Iterable[Example]) -> str:
    payload = "\n".join(row.model_dump_json() for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def split_examples(
    rows: list[Example], *, seed: int = 0, fractions: tuple[float, ...] = (0.7, 0.1, 0.1, 0.1)
) -> dict[str, list[Example]]:
    if len(fractions) != 4 or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must contain four values summing to one")
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    first_group: dict[str, int] = {}
    first_state: dict[str, int] = {}
    for i, row in enumerate(rows):
        state_key = canonical_state(row.state)
        if row.group in first_group:
            union(i, first_group[row.group])
        else:
            first_group[row.group] = i
        if state_key in first_state:
            union(i, first_state[state_key])
        else:
            first_state[state_key] = i
    buckets: dict[int, list[Example]] = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[find(i)].append(row)
    keys = list(buckets)
    random.Random(seed).shuffle(keys)
    names = ("train", "validation", "calibration", "test")
    counts = [int(len(keys) * f) for f in fractions]
    counts[-1] = len(keys) - sum(counts[:-1])
    result: dict[str, list[Example]] = {}
    cursor = 0
    for name, count in zip(names, counts):
        result[name] = [row for key in keys[cursor : cursor + count] for row in buckets[key]]
        cursor += count
    if any(not result[name] for name in names):
        raise ValueError("dataset is too small for four non-empty group-isolated splits")
    return result
