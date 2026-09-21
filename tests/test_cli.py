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
