"""Azure ML managed-online-endpoint scoring contract for Resolve AI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AGENT = None


def _project_root() -> Path:
    model_root = Path(os.environ.get("AZUREML_MODEL_DIR", "")).resolve() if os.environ.get("AZUREML_MODEL_DIR") else None
    if model_root and (model_root / "src").is_dir():
        return model_root
    return Path(__file__).resolve().parents[1]


def init() -> None:
    global AGENT
    root = _project_root()
    sys.path.insert(0, str(root / "src"))
    from customer_support_agent import CustomerSupportAgent

    AGENT = CustomerSupportAgent(root / "data" / "support_knowledge.csv")


def run(raw_data: str | dict[str, object]) -> str:
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        message = payload.get("message")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        subject = payload.get("subject", "Support request")
        if not isinstance(subject, str):
            raise ValueError("subject must be a string")
        count = payload.get("candidate_count", 3)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3:
            raise ValueError("candidate_count must be an integer between 1 and 3")
        if AGENT is None:
            init()
        result = AGENT.answer(message, subject, count)
        return json.dumps({"ok": True, "result": result.to_dict()})
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return json.dumps({"ok": False, "error": str(exc), "kind": "invalid_request"})
    except Exception:
        return json.dumps({"ok": False, "error": "The support endpoint could not complete this request.", "kind": "service_error"})

