# questionnaire/store.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_JSONL_PATH: Optional[str] = None


def init_jsonl(jsonl_path: str) -> None:
    global _JSONL_PATH
    _JSONL_PATH = jsonl_path

    folder = os.path.dirname(jsonl_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(record: Dict[str, Any]) -> None:
    if not _JSONL_PATH:
        raise RuntimeError("JSONL store not initialized. Call init_jsonl() first.")

    if "created_at_utc" not in record:
        record["created_at_utc"] = _now_utc_iso()

    line = json.dumps(record, ensure_ascii=False)
    with open(_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
