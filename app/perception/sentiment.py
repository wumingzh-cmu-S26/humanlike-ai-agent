"""Sentiment analysis with DistilBERT SST-2 — lazy-loaded singleton."""
from __future__ import annotations

import re
from functools import lru_cache
from threading import Lock
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

_NEGATIVE_LEXICON = {
    "sad", "angry", "hate", "frustrated", "tired", "exhausted", "anxious",
    "scared", "worried", "broken", "stuck", "alone", "depressed", "awful", "terrible",
}
_POSITIVE_LEXICON = {
    "happy", "great", "awesome", "love", "excited", "amazing", "wonderful",
    "fantastic", "good", "thrilled", "grateful", "proud",
}


class SentimentAnalyzer:
    _MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self) -> None:
        self._pipe = None
        self._lock = Lock()

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return
            try:
                from transformers import pipeline

                log.info("sentiment_loading", model=self._MODEL_NAME)
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model=self._MODEL_NAME,
                    truncation=True,
                )
            except Exception as e:
                log.warning("sentiment_model_unavailable", error=str(e))
                self._pipe = "lexicon"  # fall back marker

    def analyze(self, text: str) -> dict[str, Any]:
        self._ensure()
        if self._pipe == "lexicon":
            return self._lexicon_score(text)
        try:
            res = self._pipe(text[:512])[0]
            label = res["label"].lower()  # POSITIVE / NEGATIVE
            score = float(res["score"])
            mapped = "positive" if label == "positive" else "negative"
            if score < 0.6:
                mapped = "neutral"
            return {"label": mapped, "score": score, "raw": res}
        except Exception as e:
            log.warning("sentiment_predict_failed", error=str(e))
            return self._lexicon_score(text)

    def _lexicon_score(self, text: str) -> dict[str, Any]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        neg = len(words & _NEGATIVE_LEXICON)
        pos = len(words & _POSITIVE_LEXICON)
        if neg > pos:
            return {"label": "negative", "score": min(0.5 + 0.1 * neg, 0.95), "raw": "lexicon"}
        if pos > neg:
            return {"label": "positive", "score": min(0.5 + 0.1 * pos, 0.95), "raw": "lexicon"}
        return {"label": "neutral", "score": 0.5, "raw": "lexicon"}


@lru_cache
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()
