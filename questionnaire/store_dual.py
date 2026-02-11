# questionnaire/store_dual.py
from __future__ import annotations

import uuid
from typing import Any, Dict

from .store import init_jsonl, append_jsonl
from .store_sqlite import init_sqlite, save_to_sqlite


_INITIALIZED = False


def init_dual_store(
    sqlite_path: str = "./data/questionnaire/questionnaire.sqlite",
    jsonl_path: str = "./data/questionnaire/questionnaire.jsonl",
) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    init_sqlite(sqlite_path)
    init_jsonl(jsonl_path)
    _INITIALIZED = True


def save_submission(payload: Dict[str, Any]) -> str:
    new_record_id = str(uuid.uuid4())

    participant_id = str(payload.get("participant_id", "")).strip()
    if not participant_id:
        raise ValueError("participant_id is required and cannot be empty.")

    try:
        phase = int(payload["phase"])
        animation_index = int(payload["animation_index"])
        model = str(payload["model"]).strip()
    except KeyError as e:
        raise ValueError(f"Missing required field: {e}")
    except Exception as e:
        raise ValueError(f"Invalid field type: {e}")

    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2.")

    if animation_index < 1 or animation_index > 5:
        raise ValueError("animation_index must be between 1 and 5.")

    if model not in ("pretrained", "retrained"):
        raise ValueError("model must be 'pretrained' or 'retrained'.")

    responses = payload.get("responses", []) or []

    effective_record_id = save_to_sqlite(
        record_id=new_record_id,
        participant_id=participant_id,
        phase=phase,
        animation_index=animation_index,
        model=model,
        payload=payload,
        responses=responses,
    )

    record = dict(payload)
    record["record_id"] = effective_record_id
    append_jsonl(record)

    return effective_record_id

