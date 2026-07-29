from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeedbackStore:
    """Append-only JSONL feedback ledger for the local demo."""

    _lock = threading.Lock()

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, ticket_id: str, rating: int, note: str = "") -> dict[str, Any]:
        if rating not in {-1, 1}:
            raise ValueError("rating must be -1 or 1")
        record = {"ticket_id": ticket_id, "rating": rating, "note": note.strip(), "created_at": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record) + "\n")
        return record

    def summary(self) -> dict[str, float | int]:
        if not self.path.exists():
            return {"responses": 0, "positive": 0, "satisfaction": 0.0}
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        positive = sum(item["rating"] == 1 for item in records)
        return {"responses": len(records), "positive": positive, "satisfaction": round(positive / len(records), 3) if records else 0.0}
