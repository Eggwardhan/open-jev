"""Client for a local SGLang open-model Jev-compatible server."""

from __future__ import annotations
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.request import Request, urlopen
from .schema import Question


class SGLangDecisionModel:
    def __init__(
        self,
        endpoint: str,
        *,
        model_id: str = "jev-latest",
        transport: Callable[[str, bytes], bytes] | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.transport = transport

    def decide(
        self, state: str | Mapping[str, Any] | Sequence[Any], question: Question
    ) -> dict[str, Any]:
        if question.type == "choice":
            assert isinstance(question.criteria, dict)
            query = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.criteria,
            }
        elif question.type == "score":
            query = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": question.criteria,
            }
        else:
            query = {"type": "noul", "instructions": question.instructions}
        payload = json.dumps(
            {"model": self.model_id, "state": state, "questions": {"decision": query}},
            ensure_ascii=False,
        ).encode()
        if self.transport is None:
            request = Request(
                self.endpoint + "/v1/systemone",
                data=payload,
                headers={"content-type": "application/json"},
            )
            with urlopen(request, timeout=120) as response:
                raw = response.read()
        else:
            raw = self.transport(self.endpoint + "/v1/systemone", payload)
        body = json.loads(raw)
        answer = body.get("answers", {}).get("decision", body.get("decision", body))
        if not isinstance(answer, dict):
            raise ValueError("SGLang response does not contain a decision object")
        answer.setdefault("model", self.model_id)
        return answer
