import torch
from open_jev.backends import BACKENDS, BackendKind, list_backends
from open_jev.encoder_model import EncoderDecisionModel
from open_jev.schema import Question


class Tok:
    def __call__(self, texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


class Model:
    def __call__(self, **kwargs):
        return type("O", (), {"logits": torch.tensor([[3.0, 1.0]])})()


def test_catalog_contains_local_families():
    ids = {item.model_id for item in list_backends()}
    assert "Qwen/Qwen3-0.6B" in ids
    assert "heman10x/rlcd-modernbert-151m" in ids
    assert BACKENDS["Qwen/Qwen3-0.6B"].kind is BackendKind.CAUSAL


def test_encoder_backend_maps_candidate_logits():
    question = Question(
        type="choice", instructions="route", criteria={"safe": "safe", "risk": "risk"}
    )
    result = EncoderDecisionModel(Model(), Tok(), model_id="test").decide("context", question)
    assert result["label"] == "safe"
    assert set(result["probabilities"]) == {"safe", "risk"}


def test_sglang_backend_uses_local_endpoint_contract():
    from open_jev.sglang_backend import SGLangDecisionModel

    q = Question(type="noul", instructions="is this safe?")
    seen = {}

    def transport(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return b'{"answers":{"decision":{"label":"true","probabilities":{"true":0.9,"false":0.1}}}}'

    result = SGLangDecisionModel("http://127.0.0.1:30000", transport=transport).decide("x", q)
    assert result["label"] == "true"
    assert seen["url"].endswith("/v1/systemone")


def test_encoder_prompt_uses_gliclass_label_markers():
    from open_jev.encoder_model import build_encoder_prompt

    question = Question(
        type="choice", instructions="route", criteria={"safe": "safe", "risk": "risk"}
    )
    prompt = build_encoder_prompt("context", question)
    assert prompt.startswith("<<LABEL>>") and "<<SEP>>" in prompt and "Question: route" in prompt
