"""Deterministic rule-labelled cases for pipeline validation, not real-world evaluation."""

from __future__ import annotations

import itertools
import random

from .schema import Example, Question


def generate_synthetic_dataset(*, groups: int = 600, seed: int = 7) -> list[Example]:
    """Sample unique combinations; all questions about a case share its split.

    Risk and latency determine the labels. Kind, channel, region and service
    are nuisance features, sampled independently of the rules. No case ID or
    label is included in the model's state. This is an in-distribution demo.
    """
    fields = {
        "risk_band": ("low", "medium", "high"),
        "latency_band": ("fast", "normal", "slow"),
        "case_kind": ("billing", "access", "content"),
        "channel": ("email", "chat", "phone"),
        "region": ("north", "south", "east", "west"),
        "service": ("storage", "compute", "network"),
    }
    states = list(itertools.product(*fields.values()))
    if not 10 <= groups <= len(states):
        raise ValueError(f"groups must be between 10 and {len(states)}")
    rng = random.Random(seed)
    selected = rng.sample(states, groups)
    choice = Question(
        type="choice",
        instructions="Choose the operational response for this case.",
        criteria={"automate": "low risk and not slow", "review": "other cases"},
    )
    noul = Question(type="noul", instructions="Does this case require escalation?")
    score = Question(
        type="score",
        instructions="Rate the case severity from low to critical.",
        criteria=["low", "moderate", "high", "critical"],
    )
    rows = []
    for index, values in enumerate(selected):
        state = dict(zip(fields, values))
        risk, latency = state["risk_band"], state["latency_band"]
        safe = risk == "low" and latency != "slow"
        escalate = risk == "high" or latency == "slow"
        severity = 0 if risk == "low" else 1 if risk == "medium" else 3 if latency == "slow" else 2
        group = f"case-{index:04d}"
        common = {"group": group, "source": "synthetic-v2", "state": state}
        # Vary candidate order so label position cannot act as a shortcut.
        keys = list(choice.criteria)
        rng.shuffle(keys)
        shuffled_choice = choice.model_copy(
            update={"criteria": {k: choice.criteria[k] for k in keys}}
        )
        for question, label in (
            (shuffled_choice, "automate" if safe else "review"),
            (noul, escalate),
            (score, severity),
        ):
            rows.append(
                Example(id=f"{group}-{question.type}", question=question, label=label, **common)
            )
    return rows
