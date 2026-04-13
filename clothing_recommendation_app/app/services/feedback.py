from __future__ import annotations

import json
from datetime import datetime

from app.core.config import FEEDBACK_PATH
from app.schemas import FeedbackRequest


def save_feedback(payload: FeedbackRequest) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = payload.model_dump()
    record["submitted_at"] = datetime.utcnow().isoformat()

    with FEEDBACK_PATH.open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(record) + "\n")
