import torch
from open_jev.local_model import LocalDecisionModel, build_decision_prompt
from open_jev.schema import Question


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [ord(text[0])]

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}


class FakeModel:
    def __call__(self, input_ids):
        logits = torch.zeros(1, input_ids.shape[1], 128)
        logits[:, -1, 65] = 3.0
        logits[:, -1, 66] = 1.0
        return type("Output", (), {"logits": logits})()


def test_prompt_is_explicit_and_candidate_bounded():
    question = Question(
        type="choice", instructions="route this", criteria={"a": "billing", "b": "technical"}
    )
    prompt = build_decision_prompt("charged twice", question)
    assert "A: billing" in prompt and "B: technical" in prompt
    assert "Return exactly one label" in prompt


def test_local_decision_maps_letter_probabilities_to_keys():
    question = Question(
        type="choice",
        instructions="route this",
        criteria={"billing": "Payments", "technical": "Bugs"},
    )
    model = LocalDecisionModel(FakeModel(), FakeTokenizer())
    result = model.decide("charged twice", question)
    assert result["label"] == "billing"
    assert set(result["probabilities"]) == {"billing", "technical"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6


def test_auto_device_resolves_for_injected_model():
    question = Question(type="choice", instructions="route", criteria={"a": "A", "b": "B"})
    result = LocalDecisionModel(FakeModel(), FakeTokenizer(), device="auto").decide("x", question)
    assert result["label"] == "a"


def test_chat_template_is_used_when_available():
    called = []

    class ChatTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            called.append(kwargs.get("add_generation_prompt"))
            return "CHAT_TEMPLATE"

    question = Question(type="choice", instructions="route", criteria={"a": "A", "b": "B"})
    model = LocalDecisionModel(FakeModel(), ChatTokenizer())
    result = model.decide("x", question)
    assert result["label"] == "a"
    assert called == [True]
