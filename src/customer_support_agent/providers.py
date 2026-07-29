from __future__ import annotations

import json
import os
from typing import Protocol

from .models import SearchHit


class CandidateProvider(Protocol):
    def generate(self, message: str, category: str, hits: list[SearchHit], count: int) -> list[tuple[str, str]]: ...


class LocalCandidateProvider:
    def generate(self, message: str, category: str, hits: list[SearchHit], count: int = 3) -> list[tuple[str, str]]:
        policy = hits[0].article.body if hits and hits[0].score > 0 else "A specialist will inspect your case and follow up with the next steps."
        choices = [
            ("warm", f"Thanks for reaching out. I understand the concern. {policy}"),
            ("concise", f"Here is the recommended next step for this {category} request: {policy}"),
            ("action-led", f"Let's get this resolved. {policy} Reply with the requested non-sensitive details if you need more help."),
        ]
        return choices[:max(1, min(count, len(choices)))]


class OpenAICandidateProvider:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai before selecting the OpenAI provider") from exc
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, message: str, category: str, hits: list[SearchHit], count: int = 3) -> list[tuple[str, str]]:
        evidence = [{"id": h.article.id, "body": h.article.body} for h in hits]
        result = self.client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            instructions="Return JSON: {candidates:[{strategy,text}]}. Draft distinct, empathetic support replies grounded only in evidence. Never request credentials or full payment details.",
            input=json.dumps({"message": message, "category": category, "evidence": evidence, "count": count}),
        )
        data = json.loads(result.output_text)
        return [(item["strategy"], item["text"]) for item in data["candidates"][:count]]


def provider_from_environment() -> CandidateProvider:
    return OpenAICandidateProvider() if os.environ.get("AGENT_PROVIDER", "local").lower() == "openai" else LocalCandidateProvider()
