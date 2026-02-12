# questionnaire/store_sqlite.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


_DB_PATH: Optional[str] = None


def init_sqlite(db_path: str) -> None:
    global _DB_PATH
    _DB_PATH = db_path

    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questionnaire_submissions (
                record_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                animation_index INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(participant_id, phase, animation_index, model)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questionnaire_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                animation_index INTEGER NOT NULL,
                model TEXT NOT NULL,
                question_id TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(record_id) REFERENCES questionnaire_submissions(record_id) ON DELETE CASCADE
            );
            """
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_record ON questionnaire_items(record_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_participant ON questionnaire_items(participant_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_phase ON questionnaire_items(phase);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_question ON questionnaire_items(question_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_model ON questionnaire_items(model);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_submissions_participant ON questionnaire_submissions(participant_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_submissions_phase ON questionnaire_submissions(phase);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_submissions_model ON questionnaire_submissions(model);")

        conn.commit()
    finally:
        conn.close()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_existing_record_id(
    conn: sqlite3.Connection,
    participant_id: str,
    phase: int,
    animation_index: int,
    model: str,
) -> Optional[str]:
    cur = conn.execute(
        """
        SELECT record_id
        FROM questionnaire_submissions
        WHERE participant_id = ? AND phase = ? AND animation_index = ? AND model = ?
        LIMIT 1;
        """,
        (participant_id, phase, animation_index, model),
    )
    row = cur.fetchone()
    return row[0] if row else None


def save_to_sqlite(
    record_id: str,
    participant_id: str,
    phase: int,
    animation_index: int,
    model: str,
    payload: Dict[str, Any],
    responses: List[Dict[str, Any]],
) -> str:
    if not _DB_PATH:
        raise RuntimeError("SQLite store not initialized. Call init_sqlite() first.")

    created_at = _now_utc_iso()

    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys=ON;")

        # Check if submission already exists (same participant + phase + animation + model)
        existing_id = _get_existing_record_id(
            conn, participant_id, phase, animation_index, model
        )

        # If exists → reuse same record_id
        effective_record_id = existing_id or record_id

        # Ensure payload contains correct record_id
        payload_to_save = dict(payload)
        payload_to_save["record_id"] = effective_record_id
        payload_json = json.dumps(payload_to_save, ensure_ascii=False)

        # Insert or replace submission
        conn.execute(
            """
            INSERT OR REPLACE INTO questionnaire_submissions (
                record_id, participant_id, phase, animation_index, model, created_at_utc, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                effective_record_id,
                participant_id,
                phase,
                animation_index,
                model,
                created_at,
                payload_json,
            ),
        )

        # Remove old question rows (clean overwrite on re-submit)
        conn.execute(
            "DELETE FROM questionnaire_items WHERE record_id = ?;",
            (effective_record_id,),
        )

        # Insert question-wise rows
        for item in responses or []:
            qid = str(item.get("question_id", "")).strip()
            if not qid:
                continue

            ans = item.get("answer", {})
            ans_json = json.dumps(ans, ensure_ascii=False)

            conn.execute(
                """
                INSERT INTO questionnaire_items (
                    record_id, participant_id, phase, animation_index, model,
                    question_id, answer_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    effective_record_id,
                    participant_id,
                    phase,
                    animation_index,
                    model,
                    qid,
                    ans_json,
                    created_at,
                ),
            )

        conn.commit()

        # THIS IS IMPORTANT
        return effective_record_id

    finally:
        conn.close()

