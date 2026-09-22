import torch
from open_jev.calibration import TemperatureScaler, report


def test_temperature_scaler_reduces_overconfidence():
    logits = torch.tensor([[5.0, 0.0], [4.0, 0.0], [0.0, 5.0], [0.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    fitted = TemperatureScaler.fit(logits, target)
    assert fitted.temperature > 0
    assert (
        report(fitted.transform(logits), target)["nll"]
        <= report(logits.softmax(-1), target)["nll"] + 1e-5
    )


def test_cross_fit_returns_out_of_fold_probabilities():
    from open_jev.calibration import cross_fit

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    result = cross_fit(logits, target, folds=2)
    assert result.shape == logits.shape
    assert torch.isfinite(result).all()
