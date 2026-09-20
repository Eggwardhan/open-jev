import torch


def test_dynamic_choice_head_is_permutation_equivariant():
    from open_jev.model import DynamicDecisionModel

    torch.manual_seed(3)
    model = DynamicDecisionModel(vocab_size=32, hidden_size=24, max_length=8)
    state = torch.randint(0, 32, (1, 5))
    candidates = torch.randint(0, 32, (1, 3, 4))
    logits = model(state, candidates)
    permuted = model(state, candidates[:, [2, 0, 1]])
    assert torch.allclose(logits[:, [2, 0, 1]], permuted, atol=1e-6)


def test_loss_accepts_soft_targets():
    from open_jev.model import soft_cross_entropy

    logits = torch.tensor([[2.0, 0.0]])
    target = torch.tensor([[0.75, 0.25]])
    assert soft_cross_entropy(logits, target).item() > 0
