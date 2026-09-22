from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_jev.provenance import (
    build_receipt,
    file_digest,
    validate_receipt,
)


def test_build_receipt_records_relative_file_hash_and_source(tmp_path: Path) -> None:
    artifact = tmp_path / "train.jsonl"
    artifact.write_text('{"id":"a"}\n', encoding="utf-8")

    receipt = build_receipt(
        root=tmp_path,
        files=[artifact],
        sources=[
            {
                "name": "Kev",
                "url": "https://github.com/jaredpalmer/kev",
                "role": "calibration and frozen eval design",
            }
        ],
        created_at="2026-09-22T00:00:00Z",
    )

    assert receipt["format_version"] == 1
    assert receipt["files"]["train.jsonl"]["sha256"] == file_digest(artifact)
    assert receipt["sources"][0]["name"] == "Kev"


def test_validate_receipt_detects_changed_file(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.jsonl"
    artifact.write_text("one\n", encoding="utf-8")
    receipt = build_receipt(root=tmp_path, files=[artifact], created_at="now")
    assert validate_receipt(receipt, tmp_path) == []

    artifact.write_text("two\n", encoding="utf-8")
    errors = validate_receipt(receipt, tmp_path)
    assert errors == ["rows.jsonl: sha256 mismatch"]


def test_validate_receipt_rejects_missing_file_and_bad_shape(tmp_path: Path) -> None:
    receipt = {
        "format_version": 1,
        "files": {"missing.jsonl": {"sha256": "abc", "size": 3}},
    }
    assert validate_receipt(receipt, tmp_path) == ["missing.jsonl: file is missing"]
    with pytest.raises(ValueError, match="format_version"):
        validate_receipt({"files": {}}, tmp_path)


def test_receipt_is_json_serializable() -> None:
    receipt = build_receipt(root=".", files=[], sources=[], created_at="now")
    json.dumps(receipt, ensure_ascii=False)
