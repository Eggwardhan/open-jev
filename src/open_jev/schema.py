from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["choice", "noul", "score"]


class Question(BaseModel):
    """A single closed decision contract."""

    model_config = ConfigDict(extra="forbid")
    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> "Question":
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("choice criteria must be a non-empty mapping")
        if self.type == "score" and (not isinstance(self.criteria, list) or len(self.criteria) < 2):
            raise ValueError("score criteria must contain at least two levels")
        if self.type == "noul" and self.criteria is not None:
            raise ValueError("noul criteria is not used; encode the proposition in instructions")
        if isinstance(self.criteria, (dict, list)) and not self.criteria:
            raise ValueError("criteria cannot be empty")
        return self


class Example(BaseModel):
    """One training/evaluation item. Labels are intentionally task-shaped."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    group: str = Field(default="default", min_length=1)
    source: str = Field(default="unknown", min_length=1)
    state: str | dict[str, Any] | list[Any]
    question: Question
    label: Any

    @field_validator("label")
    @classmethod
    def no_nan(cls, value: Any) -> Any:
        if isinstance(value, float) and value != value:
            raise ValueError("label cannot be NaN")
        return value

    @model_validator(mode="after")
    def validate_label(self) -> "Example":
        if self.question.type == "choice":
            assert isinstance(self.question.criteria, dict)
            if not isinstance(self.label, str) or self.label not in self.question.criteria:
                raise ValueError("choice label must name one of the criteria")
        elif self.question.type == "noul":
            if isinstance(self.label, bool):
                return self
            if not isinstance(self.label, (int, float)) or not 0 <= float(self.label) <= 1:
                raise ValueError("noul label must be bool or probability in [0, 1]")
        else:
            assert isinstance(self.question.criteria, list)
            if (
                not isinstance(self.label, (int, float))
                or not 0 <= float(self.label) <= len(self.question.criteria) - 1
            ):
                raise ValueError("score label must be within score levels")
        return self

    def target(self) -> list[float]:
        if self.question.type == "choice":
            assert isinstance(self.question.criteria, dict)
            return [1.0 if key == self.label else 0.0 for key in self.question.criteria]
        if self.question.type == "noul":
            p = float(self.label)
            if isinstance(self.label, bool):
                p = float(self.label)
            return [1.0 - p, p]
        assert isinstance(self.question.criteria, list)
        levels = len(self.question.criteria)
        value = float(self.label)
        low = min(int(value), levels - 1)
        high = min(low + 1, levels - 1)
        if low == high:
            return [1.0 if i == low else 0.0 for i in range(levels)]
        result = [0.0] * levels
        result[low], result[high] = high - value, value - low
        return result
