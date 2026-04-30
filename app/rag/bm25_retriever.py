"""BM25 sparse retriever (rank_bm25)."""
from __future__ import annotations

import re
from threading import RLock
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self) -> None:
        self._lock = RLock()
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def index(self, docs: list[dict[str, Any]]) -> None:
        with self._lock:
            self._ids = [d["id"] for d in docs]
            self._tokens = [tokenize(d["text"]) for d in docs]
            self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        with self._lock:
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            ranked = sorted(
                zip(self._ids, scores, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]
