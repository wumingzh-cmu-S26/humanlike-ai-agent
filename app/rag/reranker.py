"""Cross-encoder reranker (sentence-transformers)."""
from __future__ import annotations

from threading import Lock

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model_name = get_settings().cross_encoder_model
        self._model = None
        self._lock = Lock()

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder

            log.info("cross_encoder_loading", model=self._model_name)
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, passages: list[str], top_k: int = 4) -> list[tuple[int, float]]:
        if not passages:
            return []
        self._ensure()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_k]]
