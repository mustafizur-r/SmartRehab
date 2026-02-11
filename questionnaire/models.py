# questionnaire/models.py
from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


ModelName = Literal["pretrained", "retrained"]


class ResponseItem(BaseModel):
    question_id: str = Field(..., min_length=1)
    answer: Dict[str, Any] = Field(default_factory=dict)


class SubmitQuestionnaireRequest(BaseModel):
    participant_id: str = Field(..., min_length=1, max_length=64)
    phase: int = Field(..., ge=1, le=2)
    animation_index: int = Field(..., ge=1, le=5)
    model: ModelName
    responses: List[ResponseItem] = Field(default_factory=list)


class SubmitQuestionnaireResponse(BaseModel):
    status: str = "ok"
    record_id: str = Field(..., min_length=1)
