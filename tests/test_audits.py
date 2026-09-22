import torch
from open_jev.audits import option_permutation_audit, context_shuffle_audit


def test_option_permutation_restores_labels():
    def fn(x):
        return x

    result = option_permutation_audit(
        fn, torch.tensor([[2.0, 1.0, 0.0]]), [torch.tensor([1, 0, 2])]
    )
    assert result["flip_rate"] == 0


def test_context_shuffle_reports_flip():
    def fn(x):
        return x

    result = context_shuffle_audit(fn, torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0, 2.0]]))
    assert result["flip_rate"] == 1
