from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .knowledge import configured_index, load_knowledge
from .models import AgentAnswer, Candidate
from .providers import CandidateProvider, provider_from_environment

DEFAULT_KB = Path(__file__).resolve().parents[2] / "data" / "support_knowledge.csv"
CATEGORIES = {
    "billing": ("charge", "charged", "refund", "payment", "invoice"),
    "shipping": ("shipping", "delivery", "tracking", "parcel", "late"),
    "returns": ("return", "exchange", "broken", "damaged", "replacement"),
    "account": ("login", "password", "account", "locked", "email"),
}


class CustomerSupportAgent:
    def __init__(self, knowledge_path: str | Path = DEFAULT_KB, provider: CandidateProvider | None = None):
        self.index = configured_index(load_knowledge(knowledge_path))
        self.provider = provider or provider_from_environment()

    @staticmethod
    def _classify(text: str, fallback: str) -> str:
        lowered = text.lower()
        scored = {name: sum(word in lowered for word in terms) for name, terms in CATEGORIES.items()}
        winner = max(scored, key=scored.get)
        return winner if scored[winner] else fallback

    @staticmethod
    def _sentiment(text: str) -> str:
        negative = sum(term in text.lower() for term in ("angry", "terrible", "frustrated", "unacceptable", "worst", "upset"))
        positive = sum(term in text.lower() for term in ("thanks", "please", "appreciate", "great"))
        return "negative" if negative > positive else "positive" if positive > negative else "neutral"

    def answer(self, message: str, subject: str = "Support request", candidate_count: int = 3) -> AgentAnswer:
        if len(message.strip()) < 5:
            raise ValueError("Please provide at least five characters of ticket detail")
        hits = self.index.search(f"{subject} {message}", 3)
        category = self._classify(f"{subject} {message}", hits[0].article.category)
        urgent_terms = ("fraud", "unsafe", "stolen", "legal", "breach", "injured", "chargeback")
        urgency = "urgent" if any(term in message.lower() for term in urgent_terms) else "normal"
        drafts = self.provider.generate(message, category, hits, candidate_count)
        candidates = []
        for position, (strategy, text) in enumerate(drafts):
            evidence_bonus = min(0.35, hits[0].score / 10) if hits else 0
            score = round(min(0.99, 0.58 + evidence_bonus + (0.03 if strategy == "warm" else 0) - position * 0.01), 3)
            candidates.append(Candidate(text=text, score=score, strategy=strategy))
        candidates.sort(key=lambda item: item.score, reverse=True)
        confidence = round(min(0.99, 0.5 + (hits[0].score / 8 if hits else 0)), 3)
        ticket_id = "TKT-" + hashlib.sha256(f"{subject}|{message}".encode()).hexdigest()[:8].upper()
        return AgentAnswer(
            ticket_id, category, urgency, self._sentiment(message), candidates[0].text,
            tuple(candidates), tuple(hit.article.id for hit in hits if hit.score > 0), confidence,
            urgency == "urgent" or confidence < 0.58 or bool(re.search(r"\b(password|cvv|pin)\s*[:=]", message, re.I)),
        )
