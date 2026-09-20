from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .schema import Example, Question
from .training import Trainer


class DecisionRequest(BaseModel):
    id: str = "request"
    state: str | dict | list
    question: Question


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
        return trainer.predict([example])[0]

    return app
