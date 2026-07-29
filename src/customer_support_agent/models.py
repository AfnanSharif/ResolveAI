from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Article:
    id: str
    category: str
    title: str
    body: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    article: Article
    score: float


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    strategy: str


@dataclass(frozen=True)
class AgentAnswer:
    ticket_id: str
    category: str
    urgency: str
    sentiment: str
    answer: str
    candidates: tuple[Candidate, ...]
    citations: tuple[str, ...]
    confidence: float
    human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
