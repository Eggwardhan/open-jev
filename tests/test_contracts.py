import json
import torch
from open_jev.evidence import Evidence, merge_evidence
from open_jev.replay import replay_key, ReplayReceipt
from open_jev.export import export_torchscript, verify_torchscript


def test_evidence_is_deduplicated_and_jsonable():
    result = merge_evidence(
        [Evidence(source="tool", ref="a", quote="x"), Evidence(source="tool", ref="a", quote="x")]
    )
    assert len(result) == 1
    json.dumps([item.to_dict() for item in result])


def test_replay_key_changes_with_state():
    assert replay_key({"x": 1}, "q") != replay_key({"x": 2}, "q")
    receipt = ReplayReceipt.create({"x": 1}, "q", {"choice": 0})
    assert receipt.verify()


def test_torchscript_export_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "model.pt"
    export_torchscript(model, (torch.ones(1, 2),), path)
    assert verify_torchscript(path, model, (torch.ones(1, 2),)) < 1e-5
