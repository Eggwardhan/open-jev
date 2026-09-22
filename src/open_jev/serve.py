from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .schema import Example, Question
from .training import Trainer
from .replay import replay_key


class DecisionRequest(BaseModel):
    id: str = "request"
    state: str | dict | list
    question: Question
    evidence: list[dict[str, object]] = []


def create_app(trainer: Trainer) -> FastAPI:
    app = FastAPI(title="open-jev", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/decide")
    def decide(request: DecisionRequest) -> dict[str, object]:
        example = Example(
            id=request.id,
            state=request.state,
            question=request.question,
            label=True if request.question.type == "noul" else list(request.question.criteria)[0],
        )
        result = trainer.predict([example])[0]
        result["replay_key"] = replay_key(request.state, request.question.model_dump())
        result["evidence"] = request.evidence
        return result

    return app
