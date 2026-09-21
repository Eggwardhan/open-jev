from open_jev import Example, Question
from open_jev.batching import collate
from open_jev.tokenization import WhitespaceTokenizer


def test_batch_keeps_typed_labels_separate_from_candidate_text() -> None:
    rows = [
        Example(
            id="choice",
            state="green",
            question=Question(
                type="choice", instructions="route", criteria={"go": "proceed", "stop": "halt"}
            ),
            label="go",
        ),
        Example(
            id="score",
            state="green",
            question=Question(type="score", instructions="severity", criteria=["low", "high"]),
            label=1,
        ),
    ]
    batch = collate(rows, WhitespaceTokenizer.fit(rows))
    assert batch.labels[0] == ["go", "stop"]
    assert batch.labels[1] == ["low", "high"]
    assert batch.mask.tolist() == [[True, True], [True, True]]
