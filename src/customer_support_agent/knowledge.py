from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from .models import Article, SearchHit

TOKEN = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def load_knowledge(path: str | Path) -> list[Article]:
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected = {"id", "category", "title", "body", "keywords"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"missing knowledge columns: {expected - set(reader.fieldnames or [])}")
        return [Article(row["id"], row["category"], row["title"], row["body"], tuple(filter(None, row["keywords"].split("|")))) for row in reader]


class BM25Index:
    """Compact Okapi BM25 index with no vector service dependency."""

    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75):
        if not articles:
            raise ValueError("knowledge base is empty")
        self.articles, self.k1, self.b = articles, k1, b
        self.docs = [words(f"{a.title} {a.body} {' '.join(a.keywords)}") for a in articles]
        self.avg_len = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(token for doc in self.docs for token in set(doc))
        self.idf = {term: math.log(1 + (len(self.docs) - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        query_terms = words(query)
        results = []
        for article, doc in zip(self.articles, self.docs):
            freq = Counter(doc)
            score = 0.0
            for term in query_terms:
                count = freq[term]
                norm = count + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
                score += self.idf.get(term, 0.0) * count * (self.k1 + 1) / norm if norm else 0
            results.append(SearchHit(article, round(score, 4)))
        return sorted(results, key=lambda hit: hit.score, reverse=True)[:max(1, limit)]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int = 384): self.dimensions = dimensions
    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in words(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                vector[int.from_bytes(digest, "little") % self.dimensions] += 1.0 if digest[0] & 1 else -1.0
            norm = math.sqrt(sum(item * item for item in vector)) or 1.0
            results.append([item / norm for item in vector])
        return results


class OpenAIEmbedder:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai for hosted embeddings") from exc
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), input=texts)
        return [item.embedding for item in response.data]


class FaissIndex:
    def __init__(self, articles: list[Article], embedder: Embedder | None = None):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu and numpy for FAISS retrieval") from exc
        self.articles, self.embedder, self.faiss, self.np = articles, embedder or HashingEmbedder(), faiss, np
        if not articles: raise ValueError("knowledge base is empty")
        vectors = np.asarray(self.embedder.embed([f"{a.title} {a.body} {' '.join(a.keywords)}" for a in articles]), dtype="float32")
        faiss.normalize_L2(vectors); self.index = faiss.IndexFlatIP(vectors.shape[1]); self.index.add(vectors)
    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        vector = self.np.asarray(self.embedder.embed([query]), dtype="float32"); self.faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(max(1, limit), len(self.articles)))
        return [SearchHit(self.articles[index], round(max(0.0, float(score)), 4)) for score, index in zip(scores[0], indices[0]) if index >= 0]


def configured_index(articles: list[Article]):
    if os.environ.get("AGENT_RETRIEVER", "bm25").lower() != "faiss": return BM25Index(articles)
    embedder = OpenAIEmbedder() if os.environ.get("EMBEDDING_PROVIDER", "hashing").lower() == "openai" else HashingEmbedder()
    return FaissIndex(articles, embedder)
