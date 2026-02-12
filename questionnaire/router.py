# questionnaire/router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict

from .models import SubmitQuestionnaireRequest, SubmitQuestionnaireResponse
from .schema import get_schema_by_phase, get_ground_truth_items, get_ground_truth_structured
from .store_dual import save_submission

router = APIRouter()


@router.get(
    "/schema",
    summary="Get questionnaire schema by phase (1 or 2)",
)
def get_schema(
    phase: int = Query(..., ge=1, le=2),
    animation_index: int = Query(1, ge=1, le=5),
) -> Dict[str, Any]:
    try:
        return get_schema_by_phase(
            phase=phase,
            animation_index=animation_index,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/submit",
    summary="Submit questionnaire responses (saved to sqlite + jsonl)",
    response_model=SubmitQuestionnaireResponse,
)
def submit(req: SubmitQuestionnaireRequest) -> SubmitQuestionnaireResponse:
    payload = req.model_dump()

    if req.animation_index < 1 or req.animation_index > 5:
        raise HTTPException(status_code=400, detail="animation_index must be 1..5.")

    # Store exact GT shown (important for reproducibility)
    if req.phase == 1:
        payload["ground_truth_items"] = get_ground_truth_items(req.animation_index)
        payload["ground_truth_structured"] = get_ground_truth_structured(req.animation_index)

    record_id = save_submission(payload)
    return SubmitQuestionnaireResponse(record_id=record_id)
