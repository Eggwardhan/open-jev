from open_jev.data import dataset_fingerprint, split_examples
from open_jev.synthetic import generate_synthetic_dataset


def test_synthetic_generation_is_deterministic_and_typed() -> None:
    first = generate_synthetic_dataset(groups=20, seed=11)
    second = generate_synthetic_dataset(groups=20, seed=11)
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
    assert len(first) == 60
    assert {row.question.type for row in first} == {"choice", "noul", "score"}


def test_synthetic_splits_keep_groups_isolated() -> None:
    rows = generate_synthetic_dataset(groups=40, seed=3)
    splits = split_examples(rows, seed=5)
    locations = {
        row.group: split for split, values in splits.items() for row in values for _ in [0]
    }
    assert len(locations) == 40
    assert len(set(locations.values())) == 4


def test_synthetic_has_unique_states_and_valid_rule_labels():
    from open_jev.data import canonical_state

    rows = generate_synthetic_dataset(groups=600, seed=7)
    choice_rows = [row for row in rows if row.question.type == "choice"]
    assert len({canonical_state(row.state) for row in choice_rows}) == 600
    for row in rows:
        state = row.state
        if row.question.type == "choice":
            safe = state["risk_band"] == "low" and state["latency_band"] != "slow"
            assert row.label == ("automate" if safe else "review")
        elif row.question.type == "noul":
            assert row.label == (state["risk_band"] == "high" or state["latency_band"] == "slow")
        else:
            assert row.label == (
                0
                if state["risk_band"] == "low"
                else 1
                if state["risk_band"] == "medium"
                else 3
                if state["latency_band"] == "slow"
                else 2
            )
    splits = split_examples(rows, seed=7)
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 1260,
        "validation": 180,
        "calibration": 180,
        "test": 180,
    }
    for left, left_rows in splits.items():
        for right, right_rows in splits.items():
            if left != right:
                assert {r.group for r in left_rows}.isdisjoint(r.group for r in right_rows)
                assert {canonical_state(r.state) for r in left_rows}.isdisjoint(
                    canonical_state(r.state) for r in right_rows
                )
