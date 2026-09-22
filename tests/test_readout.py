import pytest
import torch
from open_jev.readout import readout_single_token


class Tok:
    def encode(self, text, **kwargs):
        return [ord(text[0])]


class Model:
    def __call__(self, x):
        logits = torch.zeros((1, 1, 128))
        logits[0, 0, 65] = 3
        logits[0, 0, 66] = 2
        return type("R", (), {"logits": logits})()


def test_single_token_readout():
    out = readout_single_token(Model(), Tok(), "x", ["A", "B"])
    assert set(out) == {"A", "B"} and abs(sum(out.values()) - 1) < 1e-6


def test_single_token_readout_normalizes_low_precision_logits():
    class LowPrecisionModel(Model):
        def __call__(self, x):
            result = super().__call__(x)
            result.logits = result.logits.to(torch.bfloat16)
            return result

    out = readout_single_token(LowPrecisionModel(), Tok(), "x", ["A", "B"])
    assert abs(sum(out.values()) - 1) < 1e-6


def test_rejects_multitoken():
    class T(Tok):
        def encode(self, text, **kwargs):
            return [1, 2]

    with pytest.raises(ValueError):
        readout_single_token(Model(), T(), "x", ["A"])
