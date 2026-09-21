import json

from open_jev.runner import run_synthetic_training


def test_training_runner_writes_reproducible_artifacts(tmp_path) -> None:
    result = run_synthetic_training(tmp_path, groups=40, epochs=2, seed=4)
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "calibration.jsonl",
        "test.jsonl",
        "history.json",
        "metrics.json",
        "predictions.json",
        "model.pt",
    ):
        assert (tmp_path / name).exists(), name
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["test_examples"] > 0
    assert set(metrics["by_type"]) == {"choice", "noul", "score"}
    assert result["test_examples"] == metrics["test_examples"]


def test_training_runner_records_requested_device(tmp_path) -> None:
    result = run_synthetic_training(tmp_path, groups=40, epochs=1, seed=4, device="cpu")
    assert result["device"] == "cpu"


def test_seed_controls_initialization_and_training(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for destination in (first, second):
        run_synthetic_training(destination, groups=40, epochs=2, seed=8, device="cpu")
    assert (first / "history.json").read_text() == (second / "history.json").read_text()
    assert (first / "predictions.json").read_text() == (second / "predictions.json").read_text()


def test_report_contains_baselines_reload_and_hardware_evidence(tmp_path):
    result = run_synthetic_training(tmp_path, groups=40, epochs=2, seed=8, device="cpu")
    assert result["reload_max_abs_error"] <= 1e-6
    assert 1 <= result["selected_epoch"] <= 2
    assert 0 <= result["untrained"]["accuracy"] <= 1
    assert 0 <= result["majority_baseline"]["accuracy"] <= 1
    assert result["parameter_count"] > 0
    assert result["training_seconds"] > 0
    assert "score_mae" in result["by_type"]["score"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["split_fingerprints"]) == {"train", "validation", "calibration", "test"}
    assert manifest["calibration_used"] is False


def test_output_directory_cannot_silently_overwrite_previous_run(tmp_path):
    import pytest

    (tmp_path / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="empty"):
        run_synthetic_training(tmp_path, groups=40, epochs=1, device="cpu")
