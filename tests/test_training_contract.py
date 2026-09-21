import torch

from open_jev import Example, Question
from open_jev.model import DynamicDecisionModel
from open_jev.tokenization import WhitespaceTokenizer
from open_jev.training import Trainer, TrainingConfig


def test_noul_candidates_have_distinct_known_tokens():
    row = Example(
        id="noul",
        state="case",
        question=Question(type="noul", instructions="Escalate?"),
        label=True,
    )
    tokenizer = WhitespaceTokenizer.fit([row])
    assert tokenizer.encode("false") != tokenizer.encode("true")
    assert 1 not in tokenizer.encode("false true")


def test_checkpoint_reloads_architecture_without_manual_hidden_size(tmp_path):
    row = Example(
        id="noul",
        state="case",
        question=Question(type="noul", instructions="Escalate?"),
        label=True,
    )
    tokenizer = WhitespaceTokenizer.fit([row])
    trainer = Trainer(
        DynamicDecisionModel(len(tokenizer), hidden_size=24),
        tokenizer,
        TrainingConfig(device="cpu"),
    )
    path = tmp_path / "model.pt"
    trainer.save(path)
    restored = Trainer.load(path)
    assert trainer.predict([row]) == restored.predict([row])


def test_training_improves_a_balanced_noul_task():
    torch.manual_seed(5)
    rows = [
        Example(
            id=str(i),
            state="urgent urgent" if i % 2 else "routine routine",
            question=Question(type="noul", instructions="Is this urgent?"),
            label=bool(i % 2),
        )
        for i in range(32)
    ]
    tokenizer = WhitespaceTokenizer.fit(rows)
    trainer = Trainer(
        DynamicDecisionModel(len(tokenizer), hidden_size=24),
        tokenizer,
        TrainingConfig(epochs=20, learning_rate=0.02, batch_size=16, device="cpu"),
    )
    history = trainer.fit(rows)
    assert history[-1]["train_loss"] < history[0]["train_loss"] * 0.5
    predictions = trainer.predict(rows)
    assert all(
        p["label"] == ("true" if row.label else "false") for p, row in zip(predictions, rows)
    )


def test_fit_selects_best_validation_checkpoint():
    torch.manual_seed(9)
    rows = [
        Example(
            id=str(i),
            state="urgent" if i % 2 else "routine",
            question=Question(type="noul", instructions="Urgent?"),
            label=bool(i % 2),
        )
        for i in range(32)
    ]
    validation = [row.model_copy(update={"label": not row.label}) for row in rows]
    tokenizer = WhitespaceTokenizer.fit(rows)
    trainer = Trainer(
        DynamicDecisionModel(len(tokenizer), hidden_size=16),
        tokenizer,
        TrainingConfig(epochs=5, learning_rate=0.03, device="cpu"),
    )
    history = trainer.fit(rows, validation)
    from open_jev.batching import collate

    measured = trainer._loss(collate(validation, tokenizer)).item()
    assert abs(measured - min(row["validation_loss"] for row in history)) < 1e-6
