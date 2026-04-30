"""FAISS dense retriever (cosine via L2-normalized inner product)."""
from __future__ import annotations

from pathlib import Path
from threading import RLock

import faiss
import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class FaissRetriever:
    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self._lock = RLock()
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []

    def build(self, ids: list[str], vectors: list[list[float]]) -> None:
        with self._lock:
            if not ids:
                self._index = None
                self._ids = []
                return
            arr = np.asarray(vectors, dtype="float32")
            arr = _normalize(arr)
            idx = faiss.IndexFlatIP(arr.shape[1])
            idx.add(arr)
            self._index = idx
            self._ids = list(ids)
            self.dim = arr.shape[1]

    def search(self, query_vec: list[float], top_k: int = 8) -> list[tuple[str, float]]:
        with self._lock:
            if self._index is None or not self._ids:
                return []
            q = _normalize(np.asarray([query_vec], dtype="float32"))
            scores, idxs = self._index.search(q, min(top_k, len(self._ids)))
            out: list[tuple[str, float]] = []
            for score, i in zip(scores[0], idxs[0], strict=False):
                if i < 0:
                    continue
                out.append((self._ids[i], float(score)))
            return out

    def save(self, path: str | Path) -> None:
        with self._lock:
            if self._index is None:
                return
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(path))

    def load(self, path: str | Path, ids: list[str]) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with self._lock:
            try:
                self._index = faiss.read_index(str(p))
                self._ids = list(ids)
                return True
            except Exception as e:
                log.warning("faiss_load_failed", error=str(e))
                return False
