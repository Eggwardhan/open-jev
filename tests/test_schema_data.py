import pytest


def test_choice_targets_and_unknown_rejection():
    from open_jev.schema import Example, Question

    q = Question(type="choice", instructions="Route", criteria={"a": "billing", "b": "support"})
    row = Example(id="1", group="g", source="test", state="refund", question=q, label="a")
    assert row.target() == [1.0, 0.0]
    with pytest.raises(ValueError):
        Example(id="2", group="g", source="test", state="x", question=q, label="missing")


def test_noul_and_score_soft_targets():
    from open_jev.schema import Example, Question

    common = dict(id="1", group="g", source="test", state="hello")
    q = Question(type="noul", instructions="Greeting?")
    assert Example(**common, question=q, label=0.8).target() == pytest.approx([0.2, 0.8])
    q = Question(type="score", instructions="Severity?", criteria=["low", "medium", "high"])
    assert Example(**common, question=q, label=1.25).target() == [0.0, 0.75, 0.25]
    with pytest.raises(ValueError):
        Example(**common, question=q, label=3)


def test_group_and_identical_state_never_cross_splits():
    from open_jev.data import split_examples
    from open_jev.schema import Example, Question

    q = Question(type="noul", instructions="Yes?")
    rows = [
        Example(
            id=str(i), group=str(i // 2), source="test", state=f"text {i}", question=q, label=True
        )
        for i in range(80)
    ]
    rows[2] = rows[2].model_copy(update={"state": rows[0].state})
    splits = split_examples(rows, seed=42)
    assert set(splits) == {"train", "validation", "calibration", "test"}
    for key, examples in splits.items():
        assert examples
        for other, others in splits.items():
            if key != other:
                assert not {e.group for e in examples} & {e.group for e in others}
                assert not {e.state for e in examples} & {e.state for e in others}
    assert splits == split_examples(rows, seed=42)
