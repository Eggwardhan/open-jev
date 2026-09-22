from open_jev.cli import main


def test_cli_runs_synthetic_training(tmp_path, capsys):
    code = main(
        [
            "train-synthetic",
            "--output",
            str(tmp_path),
            "--groups",
            "40",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert "test_accuracy" in capsys.readouterr().out
    assert (tmp_path / "metrics.json").exists()


def test_decision_response_contains_replay_key_and_evidence():
    from open_jev.model import DynamicDecisionModel
    from open_jev.tokenization import WhitespaceTokenizer
    from open_jev.schema import Example, Question
    from open_jev.training import Trainer, TrainingConfig
    from open_jev.serve import create_app, DecisionRequest

    rows = [
        Example(
            id="x",
            state={"text": "safe"},
            question=Question(
                type="choice", instructions="pick", criteria={"a": "safe", "b": "risk"}
            ),
            label="a",
        )
    ]
    trainer = Trainer(
        DynamicDecisionModel(len(WhitespaceTokenizer.fit(rows))),
        WhitespaceTokenizer.fit(rows),
        TrainingConfig(epochs=1, device="cpu"),
    )
    app = create_app(trainer)
    result = app.routes[-1].endpoint(
        DecisionRequest(state={"text": "safe"}, question=rows[0].question, evidence=[])
    )
    assert "replay_key" in result and result["evidence"] == []


def test_cli_local_decide_accepts_qwen_model(monkeypatch, capsys):
    from open_jev import cli

    class FakeLocal:
        @classmethod
        def from_pretrained(cls, model_id, device):
            assert model_id == "Qwen/Qwen2.5-0.5B-Instruct"
            return cls()

        def decide(self, state, question):
            return {"label": "safe", "probabilities": {"safe": 1.0}}

    monkeypatch.setattr(cli, "LocalDecisionModel", FakeLocal)
    code = cli.main(
        [
            "local-decide",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--state",
            "ok",
            "--type",
            "choice",
            "--instructions",
            "route",
            "--criteria",
            '{"safe":"safe","risk":"risk"}',
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert '"label": "safe"' in capsys.readouterr().out
